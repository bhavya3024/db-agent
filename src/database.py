"""Database connection management for multiple database types."""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Protocol
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase

load_dotenv()


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    user: str = ""
    password: str = ""
    database: str = ""
    schema: str = "public"
    
    def get_connection_string(self) -> str:
        """Generate connection string based on database type."""
        if self.db_type == DatabaseType.POSTGRES:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.MYSQL:
            return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database}"
        elif self.db_type == DatabaseType.MONGODB:
            if self.user and self.password:
                return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            return f"mongodb://{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


class BaseDatabaseConnection(ABC):
    """Abstract base class for database connections."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the database connection."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        pass
    
    @abstractmethod
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table/collection."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    def is_nosql(self) -> bool:
        """Check if this is a NoSQL database."""
        return False


class SQLDatabaseConnection(BaseDatabaseConnection):
    """Manages SQL database connections using SQLAlchemy."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._engine: Optional[Engine] = None
        self._session_factory = None
    
    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            connection_string = self.config.get_connection_string()
            self._engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            self._session_factory = sessionmaker(bind=self._engine)
        return self._engine
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as a list of dictionaries."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                
                # For SELECT queries, return results
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]
                
                # For INSERT/UPDATE/DELETE, commit and return affected rows
                conn.commit()
                return [{"affected_rows": result.rowcount}]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        inspector = inspect(self.engine)
        schema_info = {
            "tables": {},
            "database": self.config.database,
            "schema": self.config.schema
        }
        
        for table_name in inspector.get_table_names(schema=self.config.schema):
            columns = inspector.get_columns(table_name, schema=self.config.schema)
            primary_keys = inspector.get_pk_constraint(table_name, schema=self.config.schema)
            foreign_keys = inspector.get_foreign_keys(table_name, schema=self.config.schema)
            
            schema_info["tables"][table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True)
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "columns": fk["constrained_columns"],
                        "references": f"{fk['referred_table']}.{fk['referred_columns']}"
                    }
                    for fk in foreign_keys
                ]
            }
        
        return schema_info
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table."""
        query = f"SELECT * FROM {self.config.schema}.{table_name} LIMIT :limit"
        return self.execute_query(query, {"limit": limit})
    
    def close(self):
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


class MongoDBConnection(BaseDatabaseConnection):
    """Manages MongoDB connections."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._client: Optional[MongoClient] = None
        self._db: Optional[MongoDatabase] = None
    
    @property
    def client(self) -> MongoClient:
        """Get or create the MongoDB client."""
        if self._client is None:
            connection_string = self.config.get_connection_string()
            self._client = MongoClient(connection_string)
            self._db = self._client[self.config.database]
        return self._client
    
    @property
    def db(self) -> MongoDatabase:
        """Get the MongoDB database."""
        if self._db is None:
            _ = self.client  # Ensure client is initialized
        return self._db
    
    def test_connection(self) -> bool:
        """Test the MongoDB connection."""
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False
    
    def is_nosql(self) -> bool:
        """MongoDB is a NoSQL database."""
        return True
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a MongoDB query. Query should be a JSON string with operation details.
        
        Supported operations:
        - {"operation": "find", "collection": "name", "filter": {}, "limit": 10}
        - {"operation": "find_one", "collection": "name", "filter": {}}
        - {"operation": "insert_one", "collection": "name", "document": {}}
        - {"operation": "insert_many", "collection": "name", "documents": []}
        - {"operation": "update_one", "collection": "name", "filter": {}, "update": {}}
        - {"operation": "update_many", "collection": "name", "filter": {}, "update": {}}
        - {"operation": "delete_one", "collection": "name", "filter": {}}
        - {"operation": "delete_many", "collection": "name", "filter": {}}
        - {"operation": "aggregate", "collection": "name", "pipeline": []}
        - {"operation": "count", "collection": "name", "filter": {}}
        """
        import json
        
        try:
            # Parse the query as JSON
            query_obj = json.loads(query)
            operation = query_obj.get("operation", "find")
            collection_name = query_obj.get("collection")
            
            if not collection_name:
                return [{"error": "Collection name is required"}]
            
            collection = self.db[collection_name]
            
            if operation == "find":
                filter_obj = query_obj.get("filter", {})
                limit = query_obj.get("limit", 100)
                projection = query_obj.get("projection")
                sort = query_obj.get("sort")
                
                cursor = collection.find(filter_obj, projection)
                if sort:
                    cursor = cursor.sort(sort)
                cursor = cursor.limit(limit)
                
                results = []
                for doc in cursor:
                    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                    results.append(doc)
                return results
            
            elif operation == "find_one":
                filter_obj = query_obj.get("filter", {})
                projection = query_obj.get("projection")
                doc = collection.find_one(filter_obj, projection)
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return [doc]
                return []
            
            elif operation == "insert_one":
                document = query_obj.get("document", {})
                result = collection.insert_one(document)
                return [{"inserted_id": str(result.inserted_id)}]
            
            elif operation == "insert_many":
                documents = query_obj.get("documents", [])
                result = collection.insert_many(documents)
                return [{"inserted_ids": [str(id) for id in result.inserted_ids]}]
            
            elif operation == "update_one":
                filter_obj = query_obj.get("filter", {})
                update = query_obj.get("update", {})
                result = collection.update_one(filter_obj, update)
                return [{"matched_count": result.matched_count, "modified_count": result.modified_count}]
            
            elif operation == "update_many":
                filter_obj = query_obj.get("filter", {})
                update = query_obj.get("update", {})
                result = collection.update_many(filter_obj, update)
                return [{"matched_count": result.matched_count, "modified_count": result.modified_count}]
            
            elif operation == "delete_one":
                filter_obj = query_obj.get("filter", {})
                result = collection.delete_one(filter_obj)
                return [{"deleted_count": result.deleted_count}]
            
            elif operation == "delete_many":
                filter_obj = query_obj.get("filter", {})
                result = collection.delete_many(filter_obj)
                return [{"deleted_count": result.deleted_count}]
            
            elif operation == "aggregate":
                pipeline = query_obj.get("pipeline", [])
                results = []
                for doc in collection.aggregate(pipeline):
                    if "_id" in doc and hasattr(doc["_id"], "__str__"):
                        doc["_id"] = str(doc["_id"])
                    results.append(doc)
                return results
            
            elif operation == "count":
                filter_obj = query_obj.get("filter", {})
                count = collection.count_documents(filter_obj)
                return [{"count": count}]
            
            else:
                return [{"error": f"Unsupported operation: {operation}"}]
                
        except json.JSONDecodeError as e:
            return [{"error": f"Invalid JSON query: {str(e)}. For MongoDB, queries must be valid JSON."}]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get MongoDB database schema information (collections and sample fields)."""
        schema_info = {
            "collections": {},
            "database": self.config.database,
            "type": "mongodb"
        }
        
        try:
            for collection_name in self.db.list_collection_names():
                collection = self.db[collection_name]
                
                # Get collection stats
                stats = self.db.command("collStats", collection_name)
                
                # Sample a document to infer fields
                sample_doc = collection.find_one()
                fields = []
                if sample_doc:
                    fields = self._infer_fields(sample_doc)
                
                # Get indexes
                indexes = []
                for index in collection.list_indexes():
                    indexes.append({
                        "name": index.get("name"),
                        "keys": list(index.get("key", {}).keys()),
                        "unique": index.get("unique", False)
                    })
                
                schema_info["collections"][collection_name] = {
                    "document_count": stats.get("count", 0),
                    "size_bytes": stats.get("size", 0),
                    "fields": fields,
                    "indexes": indexes
                }
        except Exception as e:
            schema_info["error"] = str(e)
        
        return schema_info
    
    def _infer_fields(self, doc: Dict[str, Any], prefix: str = "") -> List[Dict[str, str]]:
        """Infer field names and types from a sample document."""
        fields = []
        for key, value in doc.items():
            field_name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            field_type = type(value).__name__
            
            if field_type == "ObjectId":
                field_type = "ObjectId"
            elif field_type == "dict":
                field_type = "object"
                # Recursively get nested fields
                fields.extend(self._infer_fields(value, field_name))
            elif field_type == "list":
                field_type = "array"
                if value and isinstance(value[0], dict):
                    fields.extend(self._infer_fields(value[0], f"{field_name}[]"))
            
            fields.append({"name": field_name, "type": field_type})
        
        return fields
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample documents from a collection."""
        try:
            collection = self.db[table_name]
            results = []
            for doc in collection.find().limit(limit):
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                results.append(doc)
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


# Backwards compatibility alias
DatabaseConnection = SQLDatabaseConnection


def create_database_connection(config: DatabaseConfig) -> BaseDatabaseConnection:
    """Factory function to create the appropriate database connection."""
    if config.db_type == DatabaseType.MONGODB:
        return MongoDBConnection(config)
    else:
        return SQLDatabaseConnection(config)


class DatabaseManager:
    """Manages multiple database connections."""
    
    _instance: Optional['DatabaseManager'] = None
    
    def __init__(self):
        self.connections: Dict[str, BaseDatabaseConnection] = {}
        self.active_connection: Optional[str] = None
    
    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance
    
    def add_connection(self, name: str, config: DatabaseConfig) -> bool:
        """Add a new database connection."""
        conn = create_database_connection(config)
        if conn.test_connection():
            self.connections[name] = conn
            if self.active_connection is None:
                self.active_connection = name
            return True
        return False
    
    def get_connection(self, name: Optional[str] = None) -> Optional[BaseDatabaseConnection]:
        """Get a database connection by name or the active connection."""
        conn_name = name or self.active_connection
        if conn_name:
            return self.connections.get(conn_name)
        return None
    
    def set_active(self, name: str) -> bool:
        """Set the active database connection."""
        if name in self.connections:
            self.active_connection = name
            return True
        return False
    
    def list_connections(self) -> List[str]:
        """List all available connections."""
        return list(self.connections.keys())
    
    def close_all(self):
        """Close all database connections."""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()
        self.active_connection = None


def get_postgres_config_from_env() -> DatabaseConfig:
    """Create PostgreSQL config from environment variables."""
    return DatabaseConfig(
        db_type=DatabaseType.POSTGRES,
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DB", "postgres"),
        schema=os.getenv("POSTGRES_SCHEMA", "public")
    )


def get_mongodb_config_from_env() -> DatabaseConfig:
    """Create MongoDB config from environment variables."""
    return DatabaseConfig(
        db_type=DatabaseType.MONGODB,
        host=os.getenv("MONGODB_HOST", "localhost"),
        port=int(os.getenv("MONGODB_PORT", "27017")),
        user=os.getenv("MONGODB_USERNAME", ""),
        password=os.getenv("MONGODB_PASSWORD", ""),
        database=os.getenv("MONGODB_DATABASE", "test")
    )


def initialize_database_manager() -> DatabaseManager:
    """Initialize the database manager with connections from environment."""
    manager = DatabaseManager.get_instance()
    
    # Add PostgreSQL connection if fully configured
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_db = os.getenv("POSTGRES_DB")
    if postgres_host and postgres_db:
        postgres_config = get_postgres_config_from_env()
        if manager.add_connection("postgres", postgres_config):
            print("✓ PostgreSQL connection established")
        else:
            print("✗ Failed to connect to PostgreSQL")
    elif postgres_host:
        print("⚠ PostgreSQL: POSTGRES_DB not configured, skipping")
    
    # Add MongoDB connection if fully configured
    mongodb_host = os.getenv("MONGODB_HOST")
    mongodb_db = os.getenv("MONGODB_DATABASE")
    if mongodb_host and mongodb_db:
        mongodb_config = get_mongodb_config_from_env()
        if manager.add_connection("mongodb", mongodb_config):
            print("✓ MongoDB connection established")
        else:
            print("✗ Failed to connect to MongoDB")
    elif mongodb_host:
        print("⚠ MongoDB: MONGODB_DATABASE not configured, skipping")
    
    return manager
