from visuo_synth import DataType, Column, Table, DatabaseSchema, LangChainDataGenerationStrategy, SyntheticDataGenerator
from langchain_anthropic import ChatAnthropic

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
