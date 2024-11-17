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
        data_type_hints = {
            DataType.INTEGER: "a whole number",
            DataType.FLOAT: "a decimal number",
            DataType.STRING: "a text string",
            DataType.DATE: "a date in YYYY-MM-DD format",
            DataType.TIMESTAMP: "a timestamp in YYYY-MM-DD HH:MM:SS format",
            DataType.BOOLEAN: "true or false"
        }

        data_type_examples = {
            DataType.INTEGER: "42, 1337, 999",
            DataType.FLOAT: "42.5, 100.25, 999.99",
            DataType.STRING: "John Doe, contact@email.com, Product XYZ",
            DataType.DATE: "2024-01-15",
            DataType.TIMESTAMP: "2024-01-15 14:30:00",
            DataType.BOOLEAN: "true"
        }

        prompt = f"""Generate exactly one {data_type_hints[column.data_type]} for a column named '{column.name}'.
Respond with ONLY the value, no explanation or additional text.
The value should be in this format: {data_type_examples[column.data_type]}
"""
        if column.primary_key:
            prompt += f"\nThis is a PRIMARY KEY - it must be unique. Current values: {context.get('generated_values', {}).get(column.name, [])}"

        if context.get("table_name"):
            prompt += f"\nTable: {context['table_name']}"

        # Add more specific guidance based on the column name and context
        if "email" in column.name.lower() and column.data_type == DataType.STRING:
            prompt += "\nGenerate a valid email address"
        elif "name" in column.name.lower() and column.data_type == DataType.STRING:
            prompt += "\nGenerate a realistic person or business name"
        elif "amount" in column.name.lower() and column.data_type in [DataType.FLOAT, DataType.INTEGER]:
            prompt += "\nGenerate a reasonable monetary amount"
        elif "date" in column.name.lower() and column.data_type in [DataType.DATE, DataType.TIMESTAMP]:
            prompt += "\nGenerate a date within the last 2 years"

        return prompt

def _parse_response(self, response: str, data_type: DataType) -> Any:
    """Parse and validate LLM response based on data type."""
    if not response or not isinstance(response, str):
        raise ValueError("Invalid or empty response from LLM")

    # Remove leading/trailing whitespace
    response = response.strip()

    try:
        if data_type == DataType.INTEGER:
            # Handle cases where LLM returns formatted numbers like "1,000" or "$1000"
            cleaned = response.replace(",", "").replace("$", "").strip()
            value = int(float(cleaned))  # handle cases where LLM returns float
            return value

        elif data_type == DataType.FLOAT:
            # Handle cases where LLM returns formatted numbers like "1,000.50" or "$1000.50"
            cleaned = response.replace(",", "").replace("$", "").strip()
            return float(cleaned)

        elif data_type == DataType.BOOLEAN:
            lower_resp = response.lower()
            if lower_resp in ["true", "1", "yes", "y"]:
                return True
            elif lower_resp in ["false", "0", "no", "n"]:
                return False
            else:
                raise ValueError(f"Invalid boolean value: {response}")

        elif data_type == DataType.DATE:
            # Try multiple common date formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(response, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"Unrecognized date format: {response}")

        elif data_type == DataType.TIMESTAMP:
            # Try multiple common timestamp formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(response, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unrecognized timestamp format: {response}")

        else:  # STRING
            # Basic string validation and cleaning
            cleaned = response.strip()
            if not cleaned:
                raise ValueError("Empty string after cleaning")
            return cleaned

    except Exception as e:
        raise ValueError(
            f"Failed to parse '{response}' as {data_type.value}: {str(e)}\n"
            f"Please ensure the response matches the requested format."
        )

class SyntheticDataGenerator:
    def __init__(self, schema: DatabaseSchema, strategy: DataGenerationStrategy):
        self.schema = schema
        self.strategy = strategy
        self.generated_data = {}

    def generate(self, volumes: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate synthetic data for all tables in the schema."""
        # Reset generated data
        self.generated_data = {}

        # Get generation order based on dependencies
        generation_order = self.schema.get_generation_order()
        print(f"Table generation order: {generation_order}")

        # Validate requested volumes
        missing_tables = set(self.schema.tables.keys()) - set(volumes.keys())
        if missing_tables:
            raise ValueError(f"Volume not specified for tables: {missing_tables}")

        invalid_volumes = {table: vol for table, vol in volumes.items() if vol <= 0}
        if invalid_volumes:
            raise ValueError(f"Volume must be positive for tables: {invalid_volumes}")

        try:
            # Initialize empty storage for all tables
            for table_name in self.schema.tables:
                self.generated_data[table_name] = []

            # Generate data in correct order
            for table_name in generation_order:
                print(f"\nGenerating data for table: {table_name}")
                volume = volumes[table_name]

                # Generate data for current table
                table_data = self._generate_table_data(table_name, volume)

                if not table_data:
                    raise ValueError(f"No data was generated for table {table_name}")

                # Store generated data
                self.generated_data[table_name] = table_data

                # Validate the generated data
                self._validate_generated_data(table_name, table_data)

                print(f"✓ Successfully generated {len(table_data)} rows for {table_name}")

            print("\nData generation completed successfully")
            return self.generated_data

        except Exception as e:
            self.generated_data = {}  # Clean up on failure
            raise ValueError(f"Data generation failed: {str(e)}") from e

    def _validate_generated_data(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """Validate generated data for a table."""
        table = self.schema.tables[table_name]

        # Check all required columns are present
        for row in data:
            missing_cols = set(col.name for col in table.columns if not col.nullable) - set(row.keys())
            if missing_cols:
                raise ValueError(f"Missing required columns in {table_name}: {missing_cols}")

        # Check foreign key constraints
        for column in table.columns:
            if column.foreign_key:
                ref_table, ref_col = column.foreign_key
                ref_values = {row[ref_col] for row in self.generated_data[ref_table]}
                for row in data:
                    if row[column.name] not in ref_values:
                        raise ValueError(
                            f"Foreign key constraint violation in {table_name}.{column.name}: "
                            f"Value {row[column.name]} not found in {ref_table}.{ref_col}"
                        )

    def _generate_table_data(self, table_name: str, volume: int) -> List[Dict[str, Any]]:
        """Generate data for a single table."""
        table = self.schema.tables[table_name]
        data = []

        # Track generated values for uniqueness constraints
        unique_values = {col.name: set() for col in table.columns if col.primary_key or col.unique}

        # Pre-validate and cache foreign key references
        foreign_key_values = {}
        for column in table.columns:
            if column.foreign_key:
                ref_table, ref_col = column.foreign_key

                # Verify referenced table exists and has data
                if ref_table not in self.generated_data:
                    raise ValueError(f"Referenced table {ref_table} not found in generated data")

                ref_data = self.generated_data.get(ref_table, [])
                if not ref_data:
                    raise ValueError(
                        f"No data available in referenced table {ref_table}. "
                        f"Ensure {ref_table} is generated before {table_name}"
                    )

                # Cache available foreign key values
                ref_values = [row.get(ref_col) for row in ref_data if row.get(ref_col) is not None]
                if not ref_values:
                    raise ValueError(
                        f"No valid values found for foreign key reference "
                        f"{ref_table}.{ref_col} in table {table_name}"
                    )

                foreign_key_values[column.name] = ref_values

        print(f"\nGenerating {volume} rows for table {table_name}")

        for i in range(volume):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{volume} rows")

            try:
                row = {}
                context = {
                    "table_name": table_name,
                    "row_number": i + 1,
                    "total_rows": volume,
                    "generated_values": {k: list(v) for k, v in unique_values.items()}
                }

                # Handle foreign keys first
                for column in table.columns:
                    if column.foreign_key:
                        from random import choice
                        row[column.name] = choice(foreign_key_values[column.name])
                        context[f"ref_{column.name}"] = row[column.name]

                # Generate remaining columns
                for column in table.columns:
                    if column.name not in row:  # Skip if already handled
                        value = None
                        max_attempts = 100 if column.primary_key or column.unique else 1

                        for attempt in range(max_attempts):
                            try:
                                value = self.strategy.generate_value(column, context)

                                # Validate non-null constraint
                                if not column.nullable and value is None:
                                    raise ValueError(f"Generated NULL value for non-nullable column {column.name}")

                                # Check uniqueness constraint
                                if (column.primary_key or column.unique) and value in unique_values[column.name]:
                                    if attempt == max_attempts - 1:
                                        raise ValueError(
                                            f"Failed to generate unique value for {column.name} "
                                            f"after {max_attempts} attempts"
                                        )
                                    continue  # Try again

                                # Valid value found
                                break

                            except Exception as e:
                                if attempt == max_attempts - 1:
                                    raise ValueError(f"Failed to generate value for {column.name}: {str(e)}")

                        # Store unique values
                        if column.primary_key or column.unique:
                            unique_values[column.name].add(value)

                        row[column.name] = value
                        context[column.name] = value

                data.append(row)

            except Exception as e:
                error_msg = f"Error generating row {i + 1} for table {table_name}: {str(e)}"
                print(error_msg)
                raise ValueError(error_msg) from e

        return data

    def export_to_csv(self, output_dir: Union[str, Path]) -> None:
        """Export generated data to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for table_name, data in self.generated_data.items():
            if not data:
                continue

            file_path = output_dir / f"{table_name}.csv"
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

if __name__ == "__main__":
    # Schema definition
    customer_table = Table(
        name="customers",
        columns=[
            Column("id", DataType.INTEGER, nullable=False, primary_key=True),
            Column("name", DataType.STRING, nullable=False),
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

    try:
        # Initialize schema first
        print("Initializing schema...")
        schema = DatabaseSchema([customer_table, order_table])

        # Verify generation order
        gen_order = schema.get_generation_order()
        print(f"Generation order: {gen_order}")

        # Initialize generator
        print("Initializing data generator...")
        llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022")
        strategy = LangChainDataGenerationStrategy(llm)
        generator = SyntheticDataGenerator(schema, strategy)

        # Set up volumes
        volumes = {
            "customers": 3,  # Must be generated first
            "orders": 5
        }

        # Generate data
        print("\nStarting data generation...")
        synthetic_data = generator.generate(volumes)

        # Export to CSV
        print("\nExporting to CSV...")
        generator.export_to_csv("output_data")

        # Print samples
        print("\nGenerated Data Samples:")
        for table_name, data in synthetic_data.items():
            print(f"\n{table_name} ({len(data)} rows):")
            for i, row in enumerate(data):
                if i < 2:  # Show first 2 rows
                    print(row)

    except Exception as e:
        print(f"Error: {str(e)}")