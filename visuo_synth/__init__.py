from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import csv
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()
# from langchain_openai import ChatOpenAI

class DataType(Enum):
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    BOOLEAN = "BOOLEAN"

@dataclass
class Column:
    name: str
    data_type: DataType
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[tuple[str, str]] = None  # (table_name, column_name)
    unique: bool = False

@dataclass
class Table:
    name: str
    columns: List[Column]

class DatabaseSchema:
    def __init__(self, tables: List[Table]):
        self.tables = {table.name: table for table in tables}
        self._validate_schema()
        self._build_dependency_graph()

    def _validate_schema(self):
        """Validate schema integrity, foreign key relationships, etc."""
        # Check foreign key references
        for table in self.tables.values():
            for column in table.columns:
                if column.foreign_key:
                    ref_table, ref_col = column.foreign_key
                    if ref_table not in self.tables:
                        raise ValueError(f"Foreign key reference to non-existent table: {ref_table}")
                    ref_columns = [col.name for col in self.tables[ref_table].columns]
                    if ref_col not in ref_columns:
                        raise ValueError(f"Foreign key reference to non-existent column: {ref_col} in table {ref_table}")

    def _build_dependency_graph(self):
        """Build a graph of table dependencies based on foreign keys."""
        self.dependencies = {table: [] for table in self.tables}
        for table_name, table in self.tables.items():
            for column in table.columns:
                if column.foreign_key:
                    ref_table, _ = column.foreign_key
                    self.dependencies[table_name].append(ref_table)

    def get_generation_order(self) -> List[str]:
        """Return tables in order they should be populated (topological sort)."""
        visited = set()
        temp = set()
        order = []

        def visit(table: str):
            if table in temp:
                raise ValueError("Circular dependency detected")
            if table in visited:
                return
            temp.add(table)
            for dep in self.dependencies[table]:
                visit(dep)
            temp.remove(table)
            visited.add(table)
            order.append(table)

        for table in self.tables:
            if table not in visited:
                visit(table)

        return order[::-1]

class DataGenerationStrategy(ABC):
    @abstractmethod
    def generate_value(self, column: Column, context: Dict[str, Any]) -> Any:
        pass

class LangChainDataGenerationStrategy(DataGenerationStrategy):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.cache = {}  # Cache for similar requests

    def generate_value(self, column: Column, context: Dict[str, Any]) -> Any:
        # Generate prompt based on column and context
        prompt = self._create_prompt(column, context)

        # Check cache first
        cache_key = (column.name, str(context))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate using LLM
        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        # Parse and validate response
        value = self._parse_response(response.content, column.data_type)

        # Cache result
        self.cache[cache_key] = value
        return value

    def _create_prompt(self, column: Column, context: Dict[str, Any]) -> str:
        prompt = f"""Generate a realistic value for a {column.data_type.value} column named '{column.name}'
        Respond with ONLY the value, no explanation or additional text.
        """
        if context.get('table_name'):
            prompt += f"\nThe column is in table '{context['table_name']}'"
        if column.foreign_key:
            prompt += f"\nThis should reference {column.foreign_key[0]}.{column.foreign_key[1]}"
        return prompt

    def _parse_response(self, response: str, data_type: DataType) -> Any:
        # Convert LLM response to appropriate Python type
        try:
            if data_type == DataType.INTEGER:
                return int(response.strip())
            elif data_type == DataType.FLOAT:
                return float(response.strip())
            elif data_type == DataType.BOOLEAN:
                return response.strip().lower() == 'true'
            elif data_type == DataType.DATE:
                return datetime.strptime(response.strip(), '%Y-%m-%d').date()
            elif data_type == DataType.TIMESTAMP:
                return datetime.strptime(response.strip(), '%Y-%m-%d %H:%M:%S')
            else:  # STRING
                return response.strip()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to parse LLM response '{response}' as {data_type.value}: {str(e)}")

class SyntheticDataGenerator:
    def __init__(self, schema: DatabaseSchema, strategy: DataGenerationStrategy):
        self.schema = schema
        self.strategy = strategy
        self.generated_data = {}

    def generate(self, volumes: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate synthetic data for all tables in the schema."""
        self.generated_data = {}
        generation_order = self.schema.get_generation_order()

        # Validate requested volumes
        for table in generation_order:
            if table not in volumes:
                raise ValueError(f"Volume not specified for table: {table}")

        # Generate data in correct order
        for table_name in generation_order:
            self.generated_data[table_name] = self._generate_table_data(
                table_name,
                volumes[table_name]
            )

        return self.generated_data

    def _generate_table_data(self, table_name: str, volume: int) -> List[Dict[str, Any]]:
        """Generate data for a single table."""
        table = self.schema.tables[table_name]
        data = []

        # Track generated primary keys to ensure uniqueness
        primary_keys = set()

        for _ in range(volume):
            row = {}
            context = {"table_name": table_name}

            # First, handle foreign keys to ensure referential integrity
            for column in table.columns:
                if column.foreign_key:
                    ref_table, ref_col = column.foreign_key
                    if ref_table not in self.generated_data:
                        raise ValueError(f"Referenced table {ref_table} not yet generated")
                    # Randomly select a value from previously generated data
                    from random import choice
                    ref_data = self.generated_data[ref_table]
                    if not ref_data:
                        raise ValueError(f"No data in referenced table {ref_table}")
                    row[column.name] = choice(ref_data)[ref_col]
                    context[column.name] = row[column.name]

            # Then generate remaining columns
            for column in table.columns:
                if column.name not in row:  # Skip if already handled (foreign keys)
                    value = self.strategy.generate_value(column, context)

                    # Ensure primary key uniqueness
                    if column.primary_key:
                        attempts = 0
                        while value in primary_keys and attempts < 100:
                            value = self.strategy.generate_value(column, context)
                            attempts += 1
                        if value in primary_keys:
                            raise ValueError(f"Failed to generate unique primary key for {table_name}.{column.name}")
                        primary_keys.add(value)

                    row[column.name] = value

            data.append(row)

        return data

    def export_to_csv(self, output_dir: Union[str, Path]) -> None:
        """Export generated data to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for table_name, data in self.generated_data.items():
            if not data:
                continue

            file_path = output_dir / f"{table_name}.csv"
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

# Example usage:
if __name__ == "__main__":
    # Example schema definition
    customer_table = Table(
        name="customers",
        columns=[
            Column("id", DataType.INTEGER, nullable=False, primary_key=True),
            Column("name", DataType.STRING),
            Column("email", DataType.STRING),
            Column("created_at", DataType.TIMESTAMP)
        ]
    )

    order_table = Table(
        name="orders",
        columns=[
            Column("id", DataType.INTEGER, nullable=False, primary_key=True),
            Column("customer_id", DataType.INTEGER, foreign_key=("customers", "id")),
            Column("amount", DataType.FLOAT),
            Column("created_at", DataType.TIMESTAMP)
        ]
    )

    schema = DatabaseSchema([customer_table, order_table])

    # Initialize generator with LangChain Chat Model
    # You can use any LangChain chat model:
    # llm = ChatOpenAI(api_key="your-api-key")
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022")

    strategy = LangChainDataGenerationStrategy(llm)
    generator = SyntheticDataGenerator(schema, strategy)

    # Generate data
    volumes = {
        "customers": 100,
        "orders": 250
    }

    synthetic_data = generator.generate(volumes)

    # Export to CSV
    generator.export_to_csv("output_data")

    # Print sample of generated data
    for table_name, data in synthetic_data.items():
        print(f"\n{table_name} sample (first 3 rows):")
        for row in data[:3]:
            print(row)