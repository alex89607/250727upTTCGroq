# 🚀 TTC Benchmarking System

A comprehensive Test-Time Compute (TTC) benchmarking framework for evaluating language model optimization strategies using GroqCloud's A100x8 infrastructure.

## 🎯 Overview

This system implements and evaluates 5 core TTC algorithms with variants, providing detailed performance analysis across latency, throughput, and quality metrics. Built for production-scale evaluation with AsyncIO concurrency, real-time monitoring, and automated CI/CD pipelines.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Groq Cloud    │    │   TTC Policies   │    │   Benchmarks    │
│   A100x8        │◄──►│   (10 variants)  │◄──►│ CRMArena/Worf   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AsyncIO       │    │   Metrics        │    │   Storage       │
│   Runner        │◄──►│   Collection     │◄──►│   Parquet       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   dbt Models     │    │   GitHub        │
│   Dashboard     │◄──►│   Analytics      │◄──►│   Actions       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Quick Start

### Prerequisites

- Python 3.11+
- GroqCloud API key
- 16GB+ RAM (for local development)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd 250727upTTCGroq

# Setup environment
chmod +x setup_env.sh
./setup_env.sh

# Configure API key
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### Run Single Benchmark

```bash
# Test specific policy on specific benchmark
python runner.py --policy early_exit_28 --benchmark crmarena

# Test all benchmarks for a policy
python runner.py --policy speculative_decoding

# Verbose output
python runner.py --policy dynamic_pruning_top_p --benchmark worfbench --verbose
```

### Launch Dashboard

```bash
streamlit run streamlit/dash.py --server.port 8501
```

Visit `http://localhost:8501` for real-time performance monitoring.

## 🧠 TTC Algorithms

### Core Policies

| Algorithm | Description | Variants | Use Case |
|-----------|-------------|----------|----------|
| **Speculative Decoding** | Multi-token prediction with verification | 4-token, 6-token lookahead | High-throughput scenarios |
| **Dynamic Token Pruning** | Adaptive filtering during generation | top-p 0.9, entropy-threshold | Quality-latency balance |
| **Early Exit Blocks** | Layer-wise exit strategies | 28-layer, 32-layer exits | Ultra-low latency |
| **Adaptive KV-Cache** | Memory optimization techniques | Static, cosine decay | Memory-constrained environments |
| **Elastic Batch Repacking** | Dynamic batching strategies | Greedy, sorted bucketing | Variable load handling |

### Configuration Example

```yaml
# config.yaml
model_name: "llama-3.3-70b-versatile"
benchmarks: ["crmarena", "worfbench"]

policies:
  early_exit_28:
    max_tokens: 1024
    temperature: 0.7
    params:
      exit_layer: 28
      confidence_threshold: 0.85
      
  speculative_decoding:
    max_tokens: 1024
    temperature: 0.7
    params:
      lookahead_tokens: 4
      verification_threshold: 0.8
```

## 📊 Metrics & Evaluation

### Performance Metrics

- **first_token_latency**: Time to first token (seconds)
- **avg_token_latency**: Average per-token generation time
- **throughput**: Tokens generated per second
- **generation_time**: End-to-end response time

### Quality Metrics

- **exact_match**: Binary exact string matching
- **rouge_l**: Longest common subsequence F1 score
- **precision/recall/f1**: Standard NLP evaluation
- **semantic_similarity**: Embedding-based similarity

### Domain-Specific Metrics

- **customer_service_coverage**: CRM-specific keyword coverage
- **sales_coverage**: Sales terminology analysis
- **support_coverage**: Technical support metrics
- **retention_coverage**: Customer retention focus

## 🎛️ Dashboard Features

### Real-time Monitoring
- Live performance metrics
- Policy comparison tables
- Latency vs quality trade-offs
- Historical trend analysis

### Interactive Filters
- Policy selection (multi-select)
- Benchmark filtering
- Date range selection
- Metric threshold controls

### Visualizations
- Violin plots for latency distribution
- Scatter plots for quality trade-offs
- Time series for throughput trends
- Heatmaps for policy comparison

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The system includes a comprehensive CI/CD pipeline that:

1. **Matrix Execution**: Runs all policy variants in parallel
2. **Caching**: Caches 68GB model weights with `actions/cache@v3`
3. **Artifact Management**: Stores results and reports
4. **Automated Reporting**: Generates HTML reports and PR comments
5. **Scheduled Runs**: Daily benchmarking at 2 AM UTC

### Trigger Options

```bash
# Manual trigger with specific policy
gh workflow run ttc-benchmark.yml -f policy=early_exit_28

# Manual trigger with specific benchmark
gh workflow run ttc-benchmark.yml -f benchmark=crmarena

# Automatic triggers
git push origin main  # On push to main
# Pull requests to main
# Daily scheduled runs
```

### Artifacts

- **Individual Results**: Per-policy parquet files
- **Combined Results**: Aggregated benchmark data
- **HTML Reports**: Formatted performance summaries
- **Summary JSON**: Machine-readable metrics

## 📈 Performance Results

### Current Benchmarks (100 samples)

| Policy | Benchmark | Latency (s) | Throughput (tok/s) | ROUGE-L |
|--------|-----------|-------------|-------------------|---------|
| early_exit_28 | crmarena | 0.249 | 303.049 | 0.037 |
| speculative_decoding | worfbench | 2.471 | 221.835 | 0.073 |

### Key Insights

- **Early Exit**: Excellent latency (0.249s) but lower quality
- **Speculative Decoding**: Better quality (0.073 ROUGE-L) at higher latency cost
- **Throughput Leader**: Early Exit achieves 303 tokens/s
- **Quality Leader**: Speculative Decoding provides 2x better ROUGE-L scores

## 🛠️ Development

### Project Structure

```
250727upTTCGroq/
├── models/                 # TTC policy implementations
│   ├── __init__.py
│   └── ttc_policies.py    # Core algorithms
├── evaluation/            # Metrics and evaluation
│   ├── __init__.py
│   ├── evaluator.py      # Benchmark runner
│   └── metrics.py        # Quality metrics
├── streamlit/            # Dashboard
│   └── dash.py          # Streamlit app
├── dbt/                 # Data transformation
│   ├── models/          # SQL models
│   └── profiles.yml     # dbt configuration
├── jobs/               # Benchmark datasets
│   ├── crmarena_jobs.jsonl
│   └── worfbench_jobs.jsonl
├── .github/workflows/  # CI/CD pipelines
│   └── ttc-benchmark.yml
├── runner.py          # Main benchmark harness
├── config.yaml        # System configuration
└── requirements.txt   # Python dependencies
```

### Adding New Policies

1. **Implement Policy Class**:
```python
# models/ttc_policies.py
class MyCustomPolicy(BaseTTCPolicy):
    async def forward(self, prompt: str, policy_cfg: Dict[str, Any]) -> Tuple[str, GenerationMetrics]:
        # Your implementation here
        pass
```

2. **Register Policy**:
```python
POLICY_REGISTRY['my_custom_policy'] = MyCustomPolicy
DEFAULT_POLICY_CONFIGS['my_custom_policy'] = {
    'param1': 'default_value',
    'param2': 42
}
```

3. **Add Configuration**:
```yaml
# config.yaml
policies:
  my_custom_policy:
    max_tokens: 1024
    params:
      param1: 'custom_value'
      param2: 100
```

### Adding New Benchmarks

1. **Create JSONL Dataset**:
```jsonl
{"sample_id": "001", "prompt": "Your prompt here", "ground_truth": "Expected response"}
{"sample_id": "002", "prompt": "Another prompt", "ground_truth": "Another response"}
```

2. **Update Configuration**:
```yaml
# config.yaml
benchmarks: ["crmarena", "worfbench", "my_new_benchmark"]
```

3. **Add Metrics** (if needed):
```python
# evaluation/metrics.py
def calculate_my_custom_metric(response: str, ground_truth: str) -> float:
    # Your metric implementation
    pass
```

## 🔍 Monitoring & Alerting

### Key Performance Indicators

- **P95 Latency**: < 2.0s target
- **Throughput**: > 300 tokens/s target
- **Quality Score**: > 0.1 ROUGE-L target
- **Error Rate**: < 1% target

### Health Checks

```bash
# System health check
python test_system.py

# API connectivity test
python -c "from groq import Groq; client = Groq(); print('API OK')"

# Data pipeline test
python -c "import pandas as pd; df = pd.read_parquet('results/results.parquet'); print(f'Data OK: {len(df)} samples')"
```

## 🚀 Scaling & Production

### Infrastructure Scaling

- **Horizontal**: Multiple A100x8 instances
- **Load Balancing**: Request distribution
- **Caching**: Response caching layer
- **Edge Deployment**: Regional optimization

### Performance Optimization

- **Batch Size Tuning**: GPU utilization optimization
- **Pipeline Parallelism**: Model sharding
- **Request Batching**: Similar request grouping
- **Adaptive Routing**: Complexity-based policy selection

### Monitoring Stack

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack
- **Alerting**: PagerDuty integration
- **Tracing**: Jaeger for request tracing

## 📚 API Reference

### Runner CLI

```bash
python runner.py [OPTIONS]

Options:
  -c, --config PATH     Configuration file [default: config.yaml]
  -p, --policy TEXT     Policy name [required]
  -b, --benchmark TEXT  Benchmark name [optional]
  -o, --output PATH     Output file [default: results/results.parquet]
  -v, --verbose         Verbose output
```

### Policy Interface

```python
class BaseTTCPolicy:
    async def forward(self, prompt: str, policy_cfg: Dict[str, Any]) -> Tuple[str, GenerationMetrics]:
        """
        Generate response using TTC policy
        
        Args:
            prompt: Input prompt
            policy_cfg: Policy-specific configuration
            
        Returns:
            Tuple of (response, metrics)
        """
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints for all functions
- Include docstrings for public APIs
- Write tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GroqCloud** for A100x8 infrastructure
- **Meta** for llama-3.3-70b-versatile model
- **Streamlit** for dashboard framework
- **dbt** for data transformation
- **GitHub Actions** for CI/CD automation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**Built with ❤️ for the AI research community**

*Last updated: 2025-07-27*
