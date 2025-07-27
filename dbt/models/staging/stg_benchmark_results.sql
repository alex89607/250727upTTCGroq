{{ config(materialized='view') }}

SELECT 
    sample_id,
    policy_name,
    benchmark,
    prompt,
    response,
    ground_truth,
    timestamp,
    
    -- Latency metrics
    first_token_latency,
    avg_token_latency,
    generation_time,
    policy_overhead,
    
    -- Throughput metrics
    total_tokens,
    throughput,
    
    -- Quality metrics
    exact_match,
    precision,
    recall,
    f1,
    rouge_1,
    rouge_2,
    rouge_l,
    semantic_similarity,
    
    -- Length metrics
    prediction_length,
    reference_length,
    length_ratio,
    length_difference,
    
    -- Benchmark-specific metrics
    CASE 
        WHEN benchmark = 'crmarena' THEN customer_service_coverage
        ELSE NULL 
    END as customer_service_coverage,
    
    CASE 
        WHEN benchmark = 'crmarena' THEN sales_coverage
        ELSE NULL 
    END as sales_coverage,
    
    CASE 
        WHEN benchmark = 'crmarena' THEN support_coverage
        ELSE NULL 
    END as support_coverage,
    
    CASE 
        WHEN benchmark = 'crmarena' THEN retention_coverage
        ELSE NULL 
    END as retention_coverage,
    
    CASE 
        WHEN benchmark = 'crmarena' THEN overall_keyword_coverage
        ELSE NULL 
    END as overall_keyword_coverage,
    
    CASE 
        WHEN benchmark = 'worfbench' THEN analytical_coverage
        ELSE NULL 
    END as analytical_coverage,
    
    CASE 
        WHEN benchmark = 'worfbench' THEN structure_coverage
        ELSE NULL 
    END as structure_coverage,
    
    CASE 
        WHEN benchmark = 'worfbench' THEN reasoning_depth
        ELSE NULL 
    END as reasoning_depth,
    
    -- Derived fields
    DATE(timestamp) as run_date,
    EXTRACT(hour FROM timestamp) as run_hour,
    
    -- Policy categorization
    CASE 
        WHEN policy_name LIKE 'speculative%' THEN 'Speculative Decoding'
        WHEN policy_name LIKE 'dynamic_pruning%' THEN 'Dynamic Token Pruning'
        WHEN policy_name LIKE 'early_exit%' THEN 'Early Exit'
        WHEN policy_name LIKE 'adaptive_kv%' THEN 'Adaptive KV-Cache'
        WHEN policy_name LIKE 'elastic_batch%' THEN 'Elastic Batch Repacking'
        ELSE 'Other'
    END as policy_category,
    
    -- Performance tiers
    CASE 
        WHEN first_token_latency < 0.1 THEN 'Fast'
        WHEN first_token_latency < 0.5 THEN 'Medium'
        ELSE 'Slow'
    END as latency_tier,
    
    CASE 
        WHEN throughput > 100 THEN 'High'
        WHEN throughput > 50 THEN 'Medium'
        ELSE 'Low'
    END as throughput_tier,
    
    -- Quality tiers
    CASE 
        WHEN f1 > 0.8 THEN 'High'
        WHEN f1 > 0.6 THEN 'Medium'
        ELSE 'Low'
    END as quality_tier

FROM {{ source('raw', 'benchmark_results') }}
