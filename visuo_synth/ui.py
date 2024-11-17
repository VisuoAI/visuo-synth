import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import json
import os
from visuo_synth import (
    DataType,
    Column,
    Table,
    DatabaseSchema,
    LangChainDataGenerationStrategy,
    SyntheticDataGenerator,
)

json_parser = JsonOutputParser()


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "current_schema" not in st.session_state:
        st.session_state.current_schema = None
    if "tables" not in st.session_state:
        st.session_state.tables = []
    if "schema_json" not in st.session_state:
        st.session_state.schema_json = None
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "provider": "anthropic",
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.7,
        }


def parse_ai_schema(schema_dict):
    """Parse the AI-generated schema string into actual Table objects"""
    try:
        tables = []

        for table_name, table_def in schema_dict.items():
            columns = []
            for col_def in table_def["columns"]:
                column = Column(
                    name=col_def["name"],
                    data_type=DataType[col_def["data_type"]],
                    nullable=col_def.get("nullable", True),
                    primary_key=col_def.get("primary_key", False),
                    foreign_key=(
                        tuple(col_def["foreign_key"])
                        if col_def.get("foreign_key")
                        else None
                    ),
                    unique=col_def.get("unique", False),
                )
                columns.append(column)

            table = Table(name=table_name, columns=columns)
            tables.append(table)

        return tables
    except Exception as e:
        st.error(f"Error parsing schema: {str(e)}")
        return None


def get_llm():
    """Get configured LLM based on session state"""
    config = st.session_state.llm_config
    if config["provider"] == "anthropic":
        return ChatAnthropic(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=2000,
        )
    elif config["provider"] == "openai":
        return ChatOpenAI(
            model_name=config["model_name"], temperature=config["temperature"]
        )
    elif config["provider"] == "vertex":
        return ChatGoogleGenerativeAI(
            model_name=config["model_name"], temperature=config["temperature"]
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config['provider']}")


def generate_schema_from_description(description):
    """Use AI to generate a schema based on the description"""
    llm = get_llm()

    prompt = f"""Based on the following description, create a database schema in JSON format.
    Include appropriate data types, primary keys, foreign keys, and constraints.
    Use only these data types: INTEGER, FLOAT, STRING, DATE, TIMESTAMP, BOOLEAN
    
    Description: {description}
    
    Return only the JSON schema in this format:
    {{
        "table_name": {{
            "columns": [
                {{
                    "name": "column_name",
                    "data_type": "DATA_TYPE",
                    "nullable": boolean,
                    "primary_key": boolean,
                    "foreign_key": ["referenced_table", "referenced_column"] or null,
                    "unique": boolean
                }}
            ]
        }}
    }}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Extract JSON from response
        schema_str = response.content
        # Parse to validate JSON
        return json_parser.parse(schema_str)
    except Exception as e:
        st.error(f"Error generating schema: {str(e)}")
        return None


def display_schema_editor():
    """Display and handle the schema editor interface"""
    if not st.session_state.tables:
        st.warning("No schema loaded. Please generate or load a schema first.")
        return

    st.subheader("Schema Editor")

    modified = False
    tables_to_remove = []

    for table in st.session_state.tables:
        with st.expander(f"Table: {table.name}", expanded=True):
            # Table name editor
            new_table_name = st.text_input(
                f"Table Name", table.name, key=f"table_name_{table.name}"
            )
            if new_table_name != table.name:
                table.name = new_table_name
                modified = True

            # Delete table button
            if st.button(f"Delete Table {table.name}", key=f"del_table_{table.name}"):
                tables_to_remove.append(table)
                modified = True
                continue

            # Columns
            st.write("Columns:")
            cols_to_remove = []

            for col in table.columns:
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 1])

                with col1:
                    new_name = st.text_input(
                        "Name", col.name, key=f"col_name_{table.name}_{col.name}"
                    )
                    if new_name != col.name:
                        col.name = new_name
                        modified = True

                with col2:
                    new_type = st.selectbox(
                        "Type",
                        [dt.name for dt in DataType],
                        index=[dt.name for dt in DataType].index(col.data_type.name),
                        key=f"col_type_{table.name}_{col.name}",
                    )
                    if DataType[new_type] != col.data_type:
                        col.data_type = DataType[new_type]
                        modified = True

                with col3:
                    new_nullable = st.checkbox(
                        "Nullable",
                        col.nullable,
                        key=f"col_null_{table.name}_{col.name}",
                    )
                    if new_nullable != col.nullable:
                        col.nullable = new_nullable
                        modified = True

                with col4:
                    new_pk = st.checkbox(
                        "PK", col.primary_key, key=f"col_pk_{table.name}_{col.name}"
                    )
                    if new_pk != col.primary_key:
                        col.primary_key = new_pk
                        modified = True

                with col5:
                    new_unique = st.checkbox(
                        "Unique", col.unique, key=f"col_unique_{table.name}_{col.name}"
                    )
                    if new_unique != col.unique:
                        col.unique = new_unique
                        modified = True

                with col6:
                    if st.button("Delete", key=f"del_col_{table.name}_{col.name}"):
                        cols_to_remove.append(col)
                        modified = True

            # Remove marked columns
            for col in cols_to_remove:
                table.columns.remove(col)

            # Add new column button
            if st.button(f"Add Column to {table.name}", key=f"add_col_{table.name}"):
                table.columns.append(Column("new_column", DataType.STRING))
                modified = True

    # Remove marked tables
    for table in tables_to_remove:
        st.session_state.tables.remove(table)

    # Add new table button
    if st.button("Add New Table"):
        st.session_state.tables.append(
            Table("new_table", [Column("id", DataType.INTEGER, primary_key=True)])
        )
        modified = True

    if modified:
        try:
            # Validate schema
            schema = DatabaseSchema(st.session_state.tables)
            st.session_state.current_schema = schema
            st.success("Schema updated successfully!")
        except Exception as e:
            st.error(f"Schema validation error: {str(e)}")


def generate_synthetic_data():
    """Handle synthetic data generation"""
    if not st.session_state.current_schema:
        st.warning("Please create or load a schema first.")
        return

    st.subheader("Generate Synthetic Data")

    # Volume inputs for each table
    volumes = {}
    for table in st.session_state.tables:
        volumes[table.name] = st.number_input(
            f"Number of rows for {table.name}",
            min_value=1,
            value=10,
            key=f"volume_{table.name}",
        )

    if st.button("Generate Data"):
        try:
            with st.spinner("Generating synthetic data..."):
                llm = get_llm()
                strategy = LangChainDataGenerationStrategy(llm)
                generator = SyntheticDataGenerator(
                    st.session_state.current_schema, strategy
                )

                synthetic_data = generator.generate(volumes)
                generator.export_to_csv("output_data")

                st.success("Data generated successfully!")

                # Display sample data
                for table_name, data in synthetic_data.items():
                    st.write(f"\n{table_name} (sample of first 5 rows):")
                    if data:
                        st.dataframe(data[:5])

        except Exception as e:
            st.error(f"Error generating data: {str(e)}")


def configure_llm():
    """Configure LLM settings in sidebar"""
    with st.sidebar:
        st.header("Model Configuration")

        provider = st.selectbox(
            "Provider",
            ["anthropic", "openai", "vertex"],
            index=["anthropic", "openai", "vertex"].index(
                st.session_state.llm_config["provider"]
            ),
        )

        if provider == "anthropic":
            models = ["claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
        elif provider == "openai":
            models = ["gpt-4", "gpt-3.5-turbo"]
        else:  # google genai
            models = [""]

        model_name = st.selectbox("Model", models)
        temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.llm_config["temperature"]
        )

        # Update config if changed
        new_config = {
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature,
        }

        if new_config != st.session_state.llm_config:
            st.session_state.llm_config = new_config


def main():
    st.title("Synthetic Data Generator")

    # Initialize session state
    initialize_session_state()

    # Configure LLM
    configure_llm()

    # Step 1: Description input
    st.header("1. Describe Your Data Model")
    description = st.text_area(
        "Describe the entities and relationships you need:",
        height=150,
        placeholder="Example: I need a customer management system with customers and their orders. "
        "Each customer has a name, email, and registration date. "
        "Orders should track the amount and creation date.",
    )

    # Step 2: Generate Schema
    if description and st.button("Generate Schema"):
        with st.spinner("Generating schema from description..."):
            schema_json = generate_schema_from_description(description)
            if schema_json:
                st.session_state.schema_json = schema_json
                tables = parse_ai_schema(schema_json)
                if tables:
                    st.session_state.tables = tables
                    try:
                        st.session_state.current_schema = DatabaseSchema(tables)
                        st.success("Schema generated successfully!")
                    except Exception as e:
                        st.error(f"Error validating schema: {str(e)}")

    # Step 3: Schema Editor
    st.header("2. Edit Schema")
    display_schema_editor()

    # Step 4: Generate Data
    st.header("3. Generate Synthetic Data")
    generate_synthetic_data()


if __name__ == "__main__":
    main()
