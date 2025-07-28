import pandas as pd

# Load and check results
df = pd.read_parquet('results/results.parquet')
print(f"Total samples: {len(df)}")

# Check for valid data
valid_df = df.dropna(subset=['policy_name', 'benchmark'])
print(f"Valid samples: {len(valid_df)}")

if len(valid_df) > 0:
    print("Policies:", valid_df['policy_name'].unique())
    print("Benchmarks:", valid_df['benchmark'].unique())
    
    # Performance summary
    print("\nPerformance Summary:")
    summary = valid_df.groupby(['policy_name', 'benchmark'])[
        ['first_token_latency', 'throughput', 'rouge_l']
    ].agg(['mean', 'count']).round(3)
    print(summary)
    
    # Sample of valid data
    print("\nSample valid data:")
    print(valid_df[['policy_name', 'benchmark', 'first_token_latency', 'throughput', 'rouge_l']].head())
else:
    print("No valid data found")
    print("Sample of all data:")
    print(df[['policy_name', 'benchmark']].head(10))
