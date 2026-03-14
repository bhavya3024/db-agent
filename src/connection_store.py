"""Connection store for fetching user database connections from MongoDB."""

import os
from typing import Optional, Dict, Any
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv

from src.onepassword_resolver import get_onepassword_resolver

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
            print(f"  Available env vars: {[k for k in os.environ.keys() if 'MONGO' in k or 'CONNECTION' in k]}")
            return
        
        try:
            print(f"Connecting to connection store... (URI length: {len(mongodb_uri)})")
            
            # Determine if this is a local connection (no TLS needed)
            is_local = any(
                h in mongodb_uri
                for h in ["localhost", "127.0.0.1", "172.17.0.1", "host.docker.internal"]
            )
            
            # Build connection options
            connection_options = {
                "serverSelectionTimeoutMS": 5000,
            }
            
            # Only enable TLS for non-local connections (e.g., MongoDB Atlas)
            if not is_local:
                connection_options["tls"] = True
                connection_options["tlsAllowInvalidCertificates"] = True  # For development
                print("  Using TLS for remote connection")
            else:
                print("  Local connection detected, TLS disabled")
            
            self._client = MongoClient(mongodb_uri, **connection_options)
            
            # Extract database name from URI or use default
            db_name = os.getenv("CONNECTION_STORE_DB_NAME", "db-agent")
            self._db = self._client[db_name]
            # Test connection
            self._client.admin.command('ping')
            print(f"✓ Connection store (MongoDB) connected to database: {db_name}")
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
        if self._db is None:
            print(f"Connection store not available - cannot fetch connection {connection_id}")
            return None
        
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(connection_id):
                print(f"Invalid connection_id format: {connection_id}")
                return None
            
            print(f"Looking for connection: {connection_id} with userId: {user_id}")
            
            # Find connection with user verification
            connection = self._db.databaseconnections.find_one({
                "_id": ObjectId(connection_id),
                "userId": user_id,
            })
            
            if not connection:
                # Debug: check if connection exists at all
                any_conn = self._db.databaseconnections.find_one({"_id": ObjectId(connection_id)})
                if any_conn:
                    print(f"Connection exists but userId mismatch. Expected: {user_id}, Got: {any_conn.get('userId')}")
                else:
                    print(f"Connection {connection_id} not found in database")
                    # List all collections for debugging
                    print(f"Available collections: {self._db.list_collection_names()}")
                return None
            
            print(f"Found connection: {connection.get('name')} ({connection.get('type')})")

            connection = self.hydrate_connection_credentials(connection)
            
            # Convert ObjectId to string
            connection["_id"] = str(connection["_id"])
            
            return connection
            
        except Exception as e:
            print(f"Error fetching connection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def hydrate_connection_credentials(self, connection: Dict[str, Any]) -> Dict[str, Any]:
        """Hydrate connection credentials from 1Password when only references are stored."""
        hydrated_connection = dict(connection)

        if hydrated_connection.get("connectionString") or hydrated_connection.get("password"):
            return hydrated_connection

        resolver = get_onepassword_resolver()
        has_stored_refs = any(
            hydrated_connection.get(key)
            for key in ("connectionStringSecretRef", "passwordSecretRef", "credentialItemId")
        )

        if not has_stored_refs:
            return hydrated_connection

        if not resolver.is_configured():
            print("1Password resolver is not configured; cannot hydrate stored connection credentials")
            return hydrated_connection

        try:
            resolved_values = resolver.resolve_connection_values(hydrated_connection)
            hydrated_connection.update({key: value for key, value in resolved_values.items() if value})
        except Exception as e:
            print(
                "Failed to resolve 1Password credentials for connection "
                f"{hydrated_connection.get('_id')}: {e}"
            )

        return hydrated_connection
    
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
            from urllib.parse import quote_plus
            
            # Check if this is a MongoDB Atlas host (contains .mongodb.net)
            is_atlas = ".mongodb.net" in host
            
            if is_atlas:
                # MongoDB Atlas uses mongodb+srv:// protocol (no port)
                if username and password:
                    encoded_password = quote_plus(password)
                    return f"mongodb+srv://{username}:{encoded_password}@{host}/{database}?retryWrites=true&w=majority"
                return f"mongodb+srv://{host}/{database}?retryWrites=true&w=majority"
            else:
                # Standard MongoDB connection with port
                if username and password:
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
