@description('Location for all resources')
param location string

#disable-next-line no-unused-params
@description('Name of the environment - passed for consistency')
param environmentName string

@description('Resource token for unique naming')
param resourceToken string

@description('Tags for all resources')
param tags object

@secure()
@description('OpenAI API Key')
param openAiApiKey string

@secure()
@description('LangChain API Key')
param langChainApiKey string

@description('LangChain Project Name')
param langChainProject string

@secure()
@description('Connection Store MongoDB URI (shared with NextJS UI)')
param connectionStoreMongoDbUri string

@description('Connection Store database name')
param connectionStoreDbName string

@secure()
@description('PostgreSQL connection string for LangGraph checkpoints')
param databaseUri string

@secure()
@description('Redis connection string for LangGraph distributed locking')
param redisUri string

// ============================================================================
// User Assigned Managed Identity
// ============================================================================
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'azid${resourceToken}'
  location: location
  tags: tags
}

// ============================================================================
// Log Analytics Workspace
// ============================================================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'azlog${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ============================================================================
// Application Insights
// ============================================================================
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'azai${resourceToken}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ============================================================================
// Key Vault
// ============================================================================
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'azkv${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Key Vault Secrets Officer role for managed identity
resource kvSecretsOfficerRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, managedIdentity.id, 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================================
// Container Registry
// ============================================================================
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: 'azacr${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ACR Pull role for managed identity
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentity.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ============================================================================
// Container Apps Environment
// ============================================================================
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'azenv${resourceToken}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ============================================================================
// Container App
// ============================================================================
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'azca${resourceToken}'
  location: location
  tags: union(tags, {
    'azd-service-name': 'db-agent'
  })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: managedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'openai-api-key'
          value: openAiApiKey
        }
        {
          name: 'langchain-api-key'
          value: langChainApiKey
        }
        {
          name: 'connection-store-mongodb-uri'
          value: connectionStoreMongoDbUri
        }
        {
          name: 'database-uri'
          value: databaseUri
        }
        {
          name: 'redis-uri'
          value: redisUri
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'db-agent'
          // Use placeholder image for initial deployment, azd deploy will update it
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
            {
              name: 'LANGCHAIN_TRACING_V2'
              value: 'true'
            }
            {
              name: 'LANGCHAIN_API_KEY'
              secretRef: 'langchain-api-key'
            }
            {
              name: 'LANGCHAIN_PROJECT'
              value: langChainProject
            }
            {
              name: 'CONNECTION_STORE_MONGODB_URI'
              secretRef: 'connection-store-mongodb-uri'
            }
            {
              name: 'CONNECTION_STORE_DB_NAME'
              value: connectionStoreDbName
            }
            {
              name: 'DATABASE_URI'
              secretRef: 'database-uri'
            }
            {
              name: 'REDIS_URI'
              secretRef: 'redis-uri'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: appInsights.properties.ConnectionString
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
  dependsOn: [
    acrPullRole
    kvSecretsOfficerRole
  ]
}

// ============================================================================
// Outputs
// ============================================================================
output containerRegistryEndpoint string = containerRegistry.properties.loginServer
output containerRegistryName string = containerRegistry.name
output keyVaultName string = keyVault.name
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output managedIdentityId string = managedIdentity.id
output managedIdentityClientId string = managedIdentity.properties.clientId
