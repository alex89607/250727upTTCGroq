{{ config(materialized='table') }}

WITH policy_stats AS (
    SELECT 
        policy_name,
        policy_category,
        benchmark,
        COUNT(*) as sample_count,
        
        -- Latency statistics
        AVG(first_token_latency) as mean_first_token_latency,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY first_token_latency) as p95_first_token_latency,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY first_token_latency) as p99_first_token_latency,
        MIN(first_token_latency) as min_first_token_latency,
        MAX(first_token_latency) as max_first_token_latency,
        STDDEV(first_token_latency) as std_first_token_latency,
        
        AVG(avg_token_latency) as mean_avg_token_latency,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_token_latency) as p95_avg_token_latency,
        
        AVG(generation_time) as mean_generation_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY generation_time) as p95_generation_time,
        
        -- Throughput statistics
        AVG(throughput) as mean_throughput,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY throughput) as p95_throughput,
        MIN(throughput) as min_throughput,
        MAX(throughput) as max_throughput,
        STDDEV(throughput) as std_throughput,
        
        -- Quality statistics
        AVG(exact_match) as mean_exact_match,
        AVG(f1) as mean_f1,
        AVG(rouge_l) as mean_rouge_l,
        AVG(semantic_similarity) as mean_semantic_similarity,
        
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY f1) as p95_f1,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY f1) as p05_f1,
        
        -- Policy overhead
        AVG(policy_overhead) as mean_policy_overhead,
        
        -- Token statistics
        AVG(total_tokens) as mean_total_tokens,
        AVG(prediction_length) as mean_prediction_length,
        AVG(length_ratio) as mean_length_ratio,
        
        -- Performance distribution
        SUM(CASE WHEN latency_tier = 'Fast' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_fast_latency,
        SUM(CASE WHEN throughput_tier = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_high_throughput,
        SUM(CASE WHEN quality_tier = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_high_quality,
        
        -- Benchmark-specific metrics
        AVG(CASE WHEN benchmark = 'crmarena' THEN overall_keyword_coverage END) as mean_crm_keyword_coverage,
        AVG(CASE WHEN benchmark = 'worfbench' THEN reasoning_depth END) as mean_worf_reasoning_depth,
        
        -- Run metadata
        MIN(run_date) as first_run_date,
        MAX(run_date) as last_run_date,
        COUNT(DISTINCT run_date) as run_days
        
    FROM {{ ref('stg_benchmark_results') }}
    GROUP BY policy_name, policy_category, benchmark
),

benchmark_comparison AS (
    SELECT 
        benchmark,
        policy_category,
        
        -- Cross-policy comparisons within benchmark
        AVG(mean_first_token_latency) as benchmark_avg_latency,
        AVG(mean_throughput) as benchmark_avg_throughput,
        AVG(mean_f1) as benchmark_avg_quality,
        
        -- Best performing policies
        policy_name as best_latency_policy,
        ROW_NUMBER() OVER (PARTITION BY benchmark ORDER BY mean_first_token_latency ASC) as latency_rank,
        
        policy_name as best_throughput_policy,
        ROW_NUMBER() OVER (PARTITION BY benchmark ORDER BY mean_throughput DESC) as throughput_rank,
        
        policy_name as best_quality_policy,
        ROW_NUMBER() OVER (PARTITION BY benchmark ORDER BY mean_f1 DESC) as quality_rank
        
    FROM policy_stats
),

efficiency_metrics AS (
    SELECT 
        policy_name,
        policy_category,
        benchmark,
        
        -- Efficiency ratios
        mean_throughput / mean_first_token_latency as throughput_latency_ratio,
        mean_f1 / mean_first_token_latency as quality_latency_ratio,
        mean_f1 / mean_policy_overhead as quality_overhead_ratio,
        
        -- Composite scores (higher is better)
        (mean_f1 * mean_throughput) / (mean_first_token_latency + mean_policy_overhead) as efficiency_score,
        
        -- Pareto efficiency flags
        CASE 
            WHEN mean_first_token_latency <= PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mean_first_token_latency) OVER (PARTITION BY benchmark)
            AND mean_f1 >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mean_f1) OVER (PARTITION BY benchmark)
            THEN TRUE 
            ELSE FALSE 
        END as is_pareto_efficient
        
    FROM policy_stats
)

SELECT 
    ps.*,
    em.throughput_latency_ratio,
    em.quality_latency_ratio,
    em.quality_overhead_ratio,
    em.efficiency_score,
    em.is_pareto_efficient,
    
    -- Rankings within benchmark
    ROW_NUMBER() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_first_token_latency ASC) as latency_rank,
    ROW_NUMBER() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_throughput DESC) as throughput_rank,
    ROW_NUMBER() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_f1 DESC) as quality_rank,
    ROW_NUMBER() OVER (PARTITION BY ps.benchmark ORDER BY em.efficiency_score DESC) as efficiency_rank,
    
    -- Percentile rankings
    PERCENT_RANK() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_first_token_latency ASC) as latency_percentile,
    PERCENT_RANK() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_throughput DESC) as throughput_percentile,
    PERCENT_RANK() OVER (PARTITION BY ps.benchmark ORDER BY ps.mean_f1 DESC) as quality_percentile,
    
    -- Trade-off indicators
    CASE 
        WHEN latency_rank <= 3 AND quality_rank <= 3 THEN 'Optimal'
        WHEN latency_rank <= 3 THEN 'Fast'
        WHEN quality_rank <= 3 THEN 'High Quality'
        WHEN throughput_rank <= 3 THEN 'High Throughput'
        ELSE 'Balanced'
    END as performance_profile,
    
    CURRENT_TIMESTAMP as summary_generated_at

FROM policy_stats ps
JOIN efficiency_metrics em 
    ON ps.policy_name = em.policy_name 
    AND ps.benchmark = em.benchmark
ORDER BY ps.benchmark, em.efficiency_score DESC
