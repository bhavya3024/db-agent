"""Connection store for fetching user database connections from MongoDB."""

import os
from typing import Optional, Dict, Any
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class ConnectionStore:
    """Manages fetching database connections from the central MongoDB store."""
    
    _instance: Optional['ConnectionStore'] = None
    _client: Optional[MongoClient] = None
    _db = None
    
    def __init__(self):
        self._connect()
    
    @classmethod
    def get_instance(cls) -> 'ConnectionStore':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ConnectionStore()
        return cls._instance
    
    def _connect(self):
        """Connect to the central MongoDB store."""
        mongodb_uri = os.getenv("CONNECTION_STORE_MONGODB_URI")
        
        if not mongodb_uri:
            print("⚠ CONNECTION_STORE_MONGODB_URI not set, connection store disabled")
            return
        
        try:
            self._client = MongoClient(mongodb_uri)
            # Extract database name from URI or use default
            db_name = os.getenv("CONNECTION_STORE_DB_NAME", "dbagent")
            self._db = self._client[db_name]
            # Test connection
            self._client.admin.command('ping')
            print("✓ Connection store (MongoDB) connected")
        except Exception as e:
            print(f"✗ Failed to connect to connection store: {e}")
            self._client = None
            self._db = None
    
    def get_connection_by_id(self, connection_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a database connection by ID, verifying user ownership.
        
        Args:
            connection_id: The MongoDB ObjectId of the connection
            user_id: The user ID to verify ownership
            
        Returns:
            Connection details dict or None if not found/unauthorized
        """
        if not self._db:
            print("Connection store not available")
            return None
        
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(connection_id):
                print(f"Invalid connection_id format: {connection_id}")
                return None
            
            # Find connection with user verification
            connection = self._db.databaseconnections.find_one({
                "_id": ObjectId(connection_id),
                "userId": user_id,
            })
            
            if not connection:
                print(f"Connection not found or unauthorized: {connection_id}")
                return None
            
            # Convert ObjectId to string
            connection["_id"] = str(connection["_id"])
            
            return connection
            
        except Exception as e:
            print(f"Error fetching connection: {e}")
            return None
    
    def build_connection_string(self, connection: Dict[str, Any]) -> Optional[str]:
        """Build a connection string from connection details.
        
        Args:
            connection: Connection details from MongoDB
            
        Returns:
            Connection string or None if unable to build
        """
        db_type = connection.get("type")
        
        # If a connection string is stored, use it
        if connection.get("connectionString"):
            return connection["connectionString"]
        
        # Build connection string from individual fields
        host = connection.get("host", "localhost")
        port = connection.get("port", 5432)
        database = connection.get("database", "")
        username = connection.get("username", "")
        password = connection.get("password", "")
        
        if db_type == "postgresql":
            if username and password:
                return f"postgresql://{username}:{password}@{host}:{port}/{database}"
            return f"postgresql://{host}:{port}/{database}"
            
        elif db_type == "mongodb":
            if username and password:
                from urllib.parse import quote_plus
                encoded_password = quote_plus(password)
                return f"mongodb://{username}:{encoded_password}@{host}:{port}/{database}"
            return f"mongodb://{host}:{port}/{database}"
            
        elif db_type == "mysql":
            if username and password:
                return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            return f"mysql+pymysql://{host}:{port}/{database}"
        
        print(f"Unsupported database type: {db_type}")
        return None
    
    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


# Module-level instance getter
def get_connection_store() -> ConnectionStore:
    """Get the connection store singleton."""
    return ConnectionStore.get_instance()
