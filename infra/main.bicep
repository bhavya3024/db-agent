targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Name of the resource group')
param resourceGroupName string = 'rg-${environmentName}'

@secure()
@description('OpenAI API Key')
param openAiApiKey string

@secure()
@description('LangChain API Key')
param langChainApiKey string

@description('LangChain Project Name')
param langChainProject string = 'db-agent'

@description('PostgreSQL username')
param postgresUser string = 'postgres'

@secure()
@description('PostgreSQL password')
param postgresPassword string

@description('PostgreSQL host')
param postgresHost string = ''

@description('PostgreSQL port')
param postgresPort string = '5432'

@description('PostgreSQL database name')
param postgresDb string

@description('PostgreSQL schema')
param postgresSchema string = 'public'

@description('MongoDB username')
param mongoDbUsername string = ''

@secure()
@description('MongoDB password')
param mongoDbPassword string = ''

@description('MongoDB host')
param mongoDbHost string = ''

@description('MongoDB port')
param mongoDbPort string = '27017'

@description('MongoDB database name')
param mongoDbDatabase string

// Resource token for unique naming
var resourceToken = uniqueString(subscription().id, location, environmentName)

// Tags
var tags = {
  'azd-env-name': environmentName
}

// Resource Group
resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

// Deploy all resources in the resource group
module resources 'resources.bicep' = {
  name: 'resources-deployment'
  scope: rg
  params: {
    location: location
    environmentName: environmentName
    resourceToken: resourceToken
    tags: tags
    openAiApiKey: openAiApiKey
    langChainApiKey: langChainApiKey
    langChainProject: langChainProject
    postgresUser: postgresUser
    postgresPassword: postgresPassword
    postgresHost: postgresHost
    postgresPort: postgresPort
    postgresDb: postgresDb
    postgresSchema: postgresSchema
    mongoDbUsername: mongoDbUsername
    mongoDbPassword: mongoDbPassword
    mongoDbHost: mongoDbHost
    mongoDbPort: mongoDbPort
    mongoDbDatabase: mongoDbDatabase
  }
}

// Outputs
output RESOURCE_GROUP_ID string = rg.id
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.containerRegistryEndpoint
output AZURE_CONTAINER_REGISTRY_NAME string = resources.outputs.containerRegistryName
output AZURE_KEY_VAULT_NAME string = resources.outputs.keyVaultName
output POSTGRES_HOST string = resources.outputs.postgresHost
output COSMOS_DB_CONNECTION_STRING string = resources.outputs.cosmosDbConnectionString
output CONTAINER_APP_URL string = resources.outputs.containerAppUrl
