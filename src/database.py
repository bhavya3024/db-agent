"""Database connection management for multiple database types."""

import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"


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
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


class DatabaseConnection:
    """Manages database connections."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
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


class DatabaseManager:
    """Manages multiple database connections."""
    
    _instance: Optional['DatabaseManager'] = None
    
    def __init__(self):
        self.connections: Dict[str, DatabaseConnection] = {}
        self.active_connection: Optional[str] = None
    
    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance
    
    def add_connection(self, name: str, config: DatabaseConfig) -> bool:
        """Add a new database connection."""
        conn = DatabaseConnection(config)
        if conn.test_connection():
            self.connections[name] = conn
            if self.active_connection is None:
                self.active_connection = name
            return True
        return False
    
    def get_connection(self, name: Optional[str] = None) -> Optional[DatabaseConnection]:
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


def initialize_database_manager() -> DatabaseManager:
    """Initialize the database manager with connections from environment."""
    manager = DatabaseManager.get_instance()
    
    # Add PostgreSQL connection if configured
    if os.getenv("POSTGRES_HOST"):
        postgres_config = get_postgres_config_from_env()
        if manager.add_connection("postgres", postgres_config):
            print("✓ PostgreSQL connection established")
        else:
            print("✗ Failed to connect to PostgreSQL")
    
    return manager
