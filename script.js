// Final Updated Script - July 28, 2025

// --- DATA ---
const projectsData = [
    {
        title: "Enterprise RAG Chatbot",
        media: [
            { type: 'image', url: 'images/rag1.jpeg' },
            { type: 'image', url: 'images/rag2.jpeg' }
        ],
        description: "A production-ready RAG chatbot using LangChain, Gemini Pro, and ChromaDB for context-aware, document-grounded question answering.",
        details: [
            "Architected a full-stack RAG application using <strong>LangChain</strong> to orchestrate the retrieval and generation pipeline with <strong>Google's Gemini Pro</strong> model for both embeddings and text generation.",
            "Developed a robust <strong>FastAPI backend</strong> with Pydantic for data validation, featuring endpoints for chat, system health, and a crucial <strong>/reindex_kb</strong> endpoint to reload knowledge base documents in real-time.",
            "Engineered a flexible data ingestion module supporting multiple document formats (including <strong>.pdf, .docx, and .csv</strong>) and implemented <strong>ChromaDB</strong> as the local vector store for efficient semantic search.",
            "Implemented <strong>in-memory conversational memory</strong> to provide session-based context, enabling more natural and coherent follow-up questions from the user, with real-time source attribution for answers.",
            "Designed a clean, mobile-responsive user interface with <strong>TailwindCSS</strong> for a modern chat experience and ensured production-readiness with a fully <strong>containerized Docker environment</strong> for the backend service."
        ],
        skills: ["RAG", "LangChain", "Gemini Pro", "ChromaDB", "FastAPI", "TailwindCSS", "Docker", "Pydantic", "Vector Databases", "NLP", "REST API", "Google Generative AI"]
    },
    {
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System",
        media: [
            { type: 'image', url: 'images/ecommerce1.jpeg' },
            { type: 'image', url: 'images/ecommerce2.jpeg' }
        ],
        description: "An AI-powered platform for e-commerce analytics and sales forecasting, leveraging Facebook Prophet and interactive dashboards to drive revenue strategy and reduce stockouts.",
        details: [
            "Developed an analytics system that connects to an *optimized SQLite3 database* (ecommerce.db) with settings like PRAGMA journal_mode=WAL for better concurrency and a PRAGMA cache_size=-2000 (2MB cache) to store temporary tables in memory (PRAGMA temp_store=MEMORY). The database schema includes a sales table with order_id, customer_id, product_id, purchase_date, amount, product_category, payment_method, and shipping_country, along with a product_inventory table.",
            "Features a data generation module capable of creating *realistic historical data* for 60 days, simulating 50 customers per day, and incorporating seasonal patterns such as a weekday_factor (1.0 for weekdays, 0.7 for weekends) and hour_factors (e.g., 'morning': 0.8, 'afternoon': 1.2, 'evening': 1.5, 'night': 0.5). It generates 10 diverse products including 'Laptop' (priced at $1200), 'Smartphone' ($800), 'Headphones' ($200), 'T-shirt' ($30), 'Jeans' ($80), 'Sneakers' ($120), 'Coffee Maker' ($150), 'Backpack' ($50), 'Watch' ($300), and 'Tablet' ($600).",
            "Generates *comprehensive sales metrics* including total_revenue, average_order_value, total_orders, unique_customers, and identifies the top 5 products by revenue, as well as daily_trends for orders and revenue. For example, 'Laptop' is identified as a top product with significant revenue. This data is cached for 5 minutes (cache_timeout=300) using lru_cache to enhance performance.",
            "Performs *advanced RFM (Recency, Frequency, Monetary) analysis* by calculating additional metrics like avg_order_value, category_diversity (count of distinct product categories), and active_months (count of distinct months with purchases). Customer segmentation is performed using *K-Means clustering* with 4 defined clusters on recency, frequency, and monetary scores after applying percentile scores (1-5).",
            "Creates an interactive *Plotly dashboard* with a make_subplots layout (rows=3, cols=2, height=1200, width=1200). The dashboard includes a 'Daily Revenue Trend' (line chart), 'Customer Segments' (pie chart, e.g., showing 39.9% in one segment), 'Top Products by Revenue' (bar chart), 'Payment Method Distribution' (pie chart, e.g., 25.3% for one method), 'Hourly Sales Pattern' (line chart) based on strftime('%H', purchase_date), and 'Geographic Distribution' (bar chart) based on shipping_country.",
            "Forecasts future sales for 30 periods using the *Facebook Prophet* model, configured with yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, and specific prior scales for changepoint (0.05), seasonality (10), and holidays (10). The forecast includes ds (date), yhat (predicted value), yhat_lower, and yhat_upper for confidence intervals."
        ],
        skills: ["Streamlit", "Prophet", "Pandas", "NumPy", "Plotly Express", "Data Engineering", "Business Intelligence", "Predictive Analytics", "SQLite3", "Scikit-learn (KMeans)"]
    },
    {
        title: "AI-Powered Trading System with Risk Analytics",
        media: [
            { type: 'image', url: 'images/stocks1.jpeg' },
            { type: 'image', url: 'images/stocks2.jpeg' }
        ],
        description: "A real-time AI-driven algorithmic trading system deployed on Streamlit, providing live market data, technical indicators, and automated trade execution with robust risk management protocols.",
        details: [
            "The system initializes with a default set of symbols including 'BTC-USD', 'ETH-USD', 'AAPL', and 'GOOGL', and an initial_capital of $100,000.0. The system logs its initialization with the specified symbols.",
            "Fetches historical market data for all specified symbols using yfinance, requesting a period='60d' and interval='1h' to provide granular data for indicator calculations. Data fetching progress is logged for each symbol, including warnings if no data is received.",
            "Calculates a comprehensive set of *technical indicators* including SMA (20-period and 50-period), EMA (12-period and 26-period), MACD (MACD line, Signal line, and Histogram), RSI (14-period), and Bollinger Bands (Middle, Upper with +2*std, and Lower with -2*std). This process includes error handling for robustness.",
            "Generates trading signals based on combined conditions: a strong buy signal (value 1) when RSI < 30 and Close < BB_Lower, and a strong sell signal (value -1) when RSI > 70 and Close > BB_Upper. Additional signals (value +/- 0.5) are derived from MACD crossovers (e.g., MACD > MACD_Signal) with price confirmation against SMA_20.",
            "Implements an EnhancedRiskManager with configurable parameters such as confidence_level (0.95), max_position_size (0.2), and max_drawdown_limit (0.15). It calculates base risk metrics including VaR (np.percentile(returns, (1 - confidence_level)*100)), CVaR, Sharpe Ratio (returns.mean() / returns.std() * np.sqrt(252)), and Max Drawdown. Enhanced metrics like Sortino Ratio, Downside Risk, Calmar Ratio, and Omega Ratio are also computed.",
            "Features an AdvancedVisualizationEngine to create rich graphical reports. This includes a *correlation heatmap* of asset returns (zmin=-1, zmax=1, colorscale='RdBu'), which dynamically includes the generation date. For instance, it correctly identifies an AAPL-GOOGL correlation of 0.354 and a BTC-USD/ETH-USD correlation of 0.775.",
            "Generates a *performance dashboard* using plotly.subplots.make_subplots with a rows=2, cols=2 layout, height=1000, and width=1200. This dashboard displays 'Cumulative Returns', 'Risk Metrics' (bar charts for Sharpe, Sortino, Calmar ratios using px.colors.qualitative.Set3), 'Daily Returns Distribution' (histograms with 50 bins), and 'Rolling Volatility (20-day)' plots.",
            "Provides a detailed text report (_generate_text_report) summarizing risk analysis for each asset. For example, for BTC-USD, it reports a Sharpe Ratio of 0.036, Sortino Ratio of 0.035, Maximum Drawdown of -16.15%, Value at Risk (95%) of -0.88%, Conditional VaR of -1.47%, Calmar Ratio of 0.021, and Omega Ratio of 1.007."
        ],
        skills: ["Streamlit", "NumPy", "Pandas", "PyTorch", "Scikit-learn", "Plotly", "Alpaca API", "Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)", "Financial Analytics", "VaR", "Monte Carlo Simulation", "Real-Time Systems", "Quantitative Finance", "Time Series Forecasting", "Yfinance"]
    },
    {
        title: "AI Services Toolkit Pro (Multi-Modal AI Assistant)",
        media: [
            { type: 'image', url: 'images/toolkit.jpeg' },
            { type: 'image', url: 'images/toolkit1.jpeg' }
        ],
        description: "Architected and deployed a comprehensive, integrated Multi-Modal AI Toolkit on Hugging Face Spaces, integrating 9 state-of-the-art Transformer pipelines for diverse AI capabilities.",
        details: [
            "Features a robust application configuration (Config dataclass) with VERSION: \"1.0\", LAST_UPDATED: \"2025-02-10 18:22:33\", and flags for ENABLE_CUDA and ENABLE_TENSORRT. Key processing parameters include MAX_BATCH_SIZE: 32, MIN_BATCH_SIZE: 1, MEMORY_THRESHOLD: 0.85 (for CUDA memory optimization), CHUNK_SIZE: 640 (for image processing), MIN_CONFIDENCE: 0.25, NMS_THRESHOLD: 0.45, WHISPER_CHUNK_SIZE: 1024, and WHISPER_OVERLAP: 256. The default ONNX path is yolov5s.onnx.",
            "Includes a CUDAManager for enhanced CUDA resource management, utilizing cuda.Stream() for concurrent operations and torch.cuda.amp.GradScaler for mixed-precision training (if CUDA is enabled). It actively optimizes GPU memory by clearing the cache and collecting garbage when current_memory exceeds the MEMORY_THRESHOLD, logging the memory optimization process (e.g., 'Memory optimized Usage: 0.20%'). The system logs CUDA availability (e.g., 'CUDA Available: True') and the number/names of detected GPUs (e.g., 'GPU 0: NVIDIA A100-SXM4-40GB').",
            "Integrates *YOLOv5* for object detection, with a function to export_yolo_to_onnx() that loads yolov5s from ultralytics/yolov5, exports it to the specified ONNX path, and verifies the model. The TensorRTYOLO class then builds a TensorRT optimized engine from this ONNX model, configuring an optimization_profile to support dynamic batch sizes (min_shape=1, opt_shape=16, max_shape=32). It uses cuda.mem_alloc for efficient input/output memory allocation on GPU.",
            "Implements MultiGPUWhisper for audio processing, loading a 'tiny' Whisper model. If NUM_GPUS > 1, it utilizes torch.nn.DataParallel for parallel processing across GPUs. It processes audio by splitting it into chunks and asynchronously processing each chunk on a dedicated GPU device (cuda:i) before merging the results.",
            "Developed as a *FastAPI* application (title=\"Advanced ML Processor\", version=config.VERSION, description=\"Optimized ML processing with YOLO and Whisper\") with CORSMiddleware configured to allow all origins, credentials, methods, and headers. It exposes a /process-images endpoint that handles multiple UploadFile inputs for image inference and a /ws/audio WebSocket endpoint utilizing fastapi-websocket-pubsub for real-time audio transcription feedback and results streaming.",
            "An enhanced /health endpoint provides detailed system status, including gpu_info (device ID, name, memory used/total in MB, utilization in percent), memory_stats (recent usage, last cleanup timestamp, current threshold), and confirms if tensorrt_enabled and multi_gpu_enabled are true. It also reports the status and configuration of the 'yolo' and 'whisper' models, e.g., 'yolo' status: 'loaded', 'onnx_path': 'yolov5s.onnx', 'batch_size': 32."
        ],
        skills: ["FastAPI", "Streamlit", "Hugging Face Transformers", "PyTorch", "soundfile", "librosa", "Docker", "Full-Stack Development", "MLOps", "Whisper API", "YOLOv5", "NLP", "Speech-to-Text", "Text-to-Speech", "Real-Time Systems", "Computer Vision", "TensorRT", "ONNX", "CUDA", "WebSockets"]
    },
    {
        title: "Hybrid Predictive Maintenance System",
        media: [
            { type: 'image', url: 'images/hybrid1.jpeg' },
            { type: 'image', url: 'images/hybrid2.jpeg' }
        ],
        description: "Developed and deployed an integrated Hybrid Predictive Maintenance system on Streamlit, combining supervised learning (LSTM) and reinforcement learning for optimal maintenance recommendations.",
        details: [
            "The system is configured via SupervisedConfig, AdvancedRLConfig, and HybridSystemConfig dataclasses. Key parameters include: sequence_length=100, feature_dim=18 (though later context implies 10 features, 18 is from config) for supervised learning; state_dim=4, action_dim=4, hidden_units=256, learning_rate=0.0001, gamma=0.99, lambda_gae=0.95, clip_ratio=0.2, critic_loss_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, num_agents=4, batch_size=64, update_epochs=10, reward_scale=1.0 for RL; and update_interval=3600 seconds, prediction_horizon=24 hours, confidence_threshold=0.8, max_concurrent_predictions=10, alert_threshold=0.7, cost_threshold=50000, max_history_size=10000, visualization_enabled=True, report_format=\"detailed\" for the hybrid system.",
            "Generates synthetic data for training with num_samples=1000 and feature_dim=10. The features include vibration, temperature, pressure, current, voltage, rpm, oil_level, humidity, acoustic, and magnetic_field. The data generation simulates health_score, failure_prob, and rul as targets based on sine waves and random walks.",
            "Employs a *Multi-Agent PPO (Proximal Policy Optimization)* system, with PPOActorCritic networks for each of the 4 agents. The actor-critic network uses shared dense layers with 'relu' activation, policy layers ending in 'softmax' for action distribution, and value layers ending in a single dense unit. It selects actions using tf.random.categorical for training and tf.argmax for inference.",
            "Features a RewardModel with configurable cost_weights for maintenance (1.0), failure (10.0), and downtime (5.0). It defines specific maintenance costs (e.g., action 0: $0.0, action 1: $100.0, action 2: $500.0, action 3: $1000.0) and downtime costs (e.g., action 0: $0.0, action 1: $2.0, action 2: $4.0, action 3: $8.0).",
            "Predicts machine health metrics (health_score, failure_prob, rul) from sensor data, with an example output for a monitored machine (Machine 0) showing a health score of 0.612, failure probability of 0.388, and RUL of 61.2. The prediction for a given machine also includes a maintenance action (e.g., Action: 1 for Machine 0) and a value estimate (e.g., 0.057 for Machine 0).",
            "Incorporates an ExplainabilityModule (conceptualized with *SHAP and LIME*) that provides insights into predictions by highlighting top contributing factors based on randomly generated importance scores in the example. For Machine 0, the top factors are magnetic_field: 0.258, temperature: 0.182, and humidity: 0.157.",
            "Visualizes results through Matplotlib/Seaborn plots for 'Health Metrics', 'Maintenance Decisions' (scatterplot), 'Maintenance Costs' (line plot), and 'Feature Importance' (bar plot). These visualizations are saved to the model_checkpoint_dir (e.g., /content/hybrid_checkpoints/visualization_0.png). The system also saves final metrics to system_metrics.json."
        ],
        skills: ["Streamlit", "TensorFlow", "Keras", "NumPy", "Pandas", "SQLite3", "Deep Learning", "Reinforcement Learning (PPO)", "SHAP", "LIME", "Plotly", "Python", "Data Analytics", "Anomaly Detection", "Multi-Agent Systems"]
    },
    {
        title: "Customer Churn Prediction and API Deployment",
        media: [
            { type: 'image', url: 'images/churn1.jpeg' },
            { type: 'image', url: 'images/churn2.jpeg' }
        ],
        description: "Architected and deployed an integrated Customer Churn Prediction system on Streamlit with a FastAPI backend for model inference, achieving high accuracy and efficient real-time predictions.",
        details: [
            "Generates a synthetic customer churn dataset containing n_samples=2000 entries, with features such as age, gender, income, contract_type, monthly_charges, total_services, contacts_count, and complaints_count. Churn probability is engineered based on rules like 0.1 * (age > 50), 0.2 * (monthly_charges > 100), 0.3 * (total_services < 2), and 0.2 * (complaints_count > 2).",
            "Performs robust data preprocessing using ColumnTransformer to apply StandardScaler for numerical features (age, income, monthly_charges, total_services, contacts_count, complaints_count) and OneHotEncoder (with handle_unknown=\"ignore\" and sparse_output=False) for categorical features (gender, contract_type).",
            "Addresses class imbalance by applying *SMOTE (Synthetic Minority Over-sampling Technique)* with random_state=42 to the processed data before splitting into training and testing sets (test_size=0.2).",
            "Trains and evaluates two distinct machine learning models: a *Random Forest Classifier* (n_estimators=100, random_state=42) and a *Neural Network* built with *TensorFlow/Keras*. The Neural Network architecture includes Dense(64, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(32, activation='relu'), and a final Dense(1, activation='sigmoid') layer. It's compiled with Adam(0.001) optimizer, binary_crossentropy loss, and trained for 50 epochs with a batch size of 32.",
            "Evaluates model performance comprehensively, calculating rf_accuracy and nn_accuracy on the test set (e.g., Random Forest Accuracy: 0.9416, Neural Network Accuracy: 0.9500 in example output). Visualizations generated and saved to the churn_project/ directory include 'Feature Importances' (e.g., 'monthly_charges' 0.23, 'total_services' 0.21, 'income' 0.17), 'Confusion Matrices' for both models, 'ROC Curves Comparison' with AUC scores (e.g., Random Forest AUC=0.949, Neural Network AUC=0.957), and an 'Impact of Classification Threshold on Different Metrics' (accuracy, precision, recall, f1 for thresholds 0.1 to 0.9).",
            "Deploys a high-performance *RESTful API* using *FastAPI* and *Uvicorn*, served at http://localhost:8000/predict/. The /predict/ endpoint accepts PredictionInput (a Pydantic model with features as a dictionary) and returns timestamp, user (e.g., '03ADY'), random_forest_prediction, neural_network_prediction, and their respective probabilities (rf_probability, nn_probability). Example test cases (high_risk_customer, low_risk_customer, medium_risk_customer) with detailed feature sets are included to demonstrate API functionality and are visualized in a 'Churn Predictions by Model and Customer Profile' bar chart. All test results are saved to api_test_results.json."
        ],
        skills: ["FastAPI", "Streamlit", "Scikit-learn", "Pandas", "NumPy", "imblearn (SMOTE)", "Random Forest", "Neural Networks", "MLOps", "Model Deployment", "REST API", "Python", "Classification", "TensorFlow", "Uvicorn"]
    }
];

const playgroundAppsData = [
    {
        title: "Enterprise RAG Chatbot Demo",
        description: "An interactive demo of the RAG chatbot. Upload your own documents or use the default knowledge base to ask questions and get context-aware answers from the AI.",
        url: "https://rag-chatbot-roan-eight.vercel.app/",
        image: "images/rag2.jpeg"
    },
    {
        title: "AI-Powered Customer Churn Prediction",
        description: "An interactive Streamlit application demonstrating a machine learning model that predicts customer churn, allowing users to input customer data and see real-time predictions.",
        url: "https://futureml02-jg9jkjmv3xahqnceylr8eu.streamlit.app/",
        image: "images/churn2.jpeg"
    },
    {
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System",
        description: "A live Streamlit dashboard providing comprehensive e-commerce analytics, including sales trends, customer segmentation (RFM), and interactive visualizations for business insights.",
        url: "https://futureml01-j2lihzs8qwk6ombkpeczut.streamlit.app/",
        image: "images/ecommerce2.jpeg"
    },
    {
        title: "AI Services Toolkit Pro",
        description: "Explore a suite of multi-modal AI models, including sentiment analysis, summarization, and image captioning, all served through a user-friendly interface.",
        url: "https://ai-toolkit-nj89aumds7l6rpjas7486m.streamlit.app/",
        image: "images/toolkit1.jpeg"
    },
    {
        title: "Hybrid Predictive Maintenance Dashboard",
        description: "An interactive Streamlit dashboard showcasing real-time predictive maintenance insights, including anomaly detection and remaining useful life (RUL) predictions for industrial equipment.",
        url: "https://smart-predictive-maintenance-en77oylapplyfegbhuzf3fy.streamlit.app/Live_Dashboard",
        image: "images/hybrid2.jpeg"
    },
    {
        title: "AI-Powered Live Trading System",
        description: "An interactive Streamlit application demonstrating a deep learning-based trading system with real-time market data, technical indicators, and robust risk management.",
        url: "https://smart-predictive-maintenance-am3fyapk9yqcujd87tjcux.streamlit.app/",
        image: "images/stocks2.jpeg"
    }
];

const skillsData = [
    { title: 'Languages & Core', skills: ['Python', 'TypeScript', 'JavaScript', 'SQL', 'FastAPI', 'Flask', 'Node.js', 'Express'] },
    { title: 'AWS Cloud & Serverless', skills: ['Amplify Gen 2', 'CDK', 'Lambda', 'App Runner', 'DynamoDB', 'Cognito', 'SES', 'S3', 'ECR', 'CodeBuild', 'EventBridge', 'CloudWatch'] },
    { title: 'Frontend & 3D', skills: ['React 18', 'Vite', 'Tailwind CSS', 'Three.js', 'React-Three-Fiber', 'Zustand', 'TanStack Query', 'shadcn/Radix UI'] },
    { title: 'Generative AI & RAG', skills: ['Google Gemini', 'LangChain', 'FAISS', 'OpenAI API', 'Hugging Face', 'Prompt Engineering', 'RAG Pipelines'] },
    { title: 'Programming Languages & Frameworks', skills: ['Python (NumPy, Pandas)', 'SQL', 'FastAPI', 'Flask', 'Data Structures', 'Algorithms'] },
    { title: 'Machine Learning', skills: ['Scikit-learn', 'XGBoost', 'Random Forests', 'SVMs', 'Regression', 'Classification', 'Clustering', 'Feature Engineering', 'Model Evaluation', 'SMOTE', 'Reinforcement Learning'] },
    { title: 'Deep Learning', skills: ['TensorFlow', 'PyTorch', 'Keras', 'CNNs', 'RNNs (LSTMs, GRUs)', 'Transformers', 'Transfer Learning', 'GANs'] },
    { title: 'MLOps & Deployment', skills: ['Docker', 'Kubernetes (basics)', 'Render', 'Heroku', 'Git', 'Uvicorn', 'RESTful API', 'Microservices', 'CI/CD', 'Model Versioning', 'Monitoring', 'Hugging Face Spaces'] },
    { title: 'Big Data & Databases', skills: ['Hadoop', 'Spark', 'Apache Kafka', 'PostgreSQL', 'MySQL', 'SQLAlchemy', 'MongoDB', 'SQLite3'] },
    { title: 'Data Visualization & BI', skills: ['Plotly', 'Matplotlib', 'Seaborn', 'Dash', 'Tableau (conceptual)', 'Streamlit'] },
    { title: 'Specialized Tools', skills: ['SHAP', 'LIME', 'Prophet', 'Whisper API', 'YOLOv5', 'OpenAI API', 'LLMs (ChatGPT)', 'Financial Modeling', 'Monte Carlo Simulation', 'A/B Testing', 'Alpaca API', 'Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)'] }
];

// --- UI LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    if (!window.location.hash) {
        window.scrollTo(0, 0);
    }
    // Function to create interactive covers (NO LONGER USED FOR PROJECT CARDS DIRECTLY, but kept as a fallback/example if needed elsewhere)
    const createInteractiveCover = (project) => {
        const { type } = project.interactive_cover || {}; 
        let svgContent = '';
        switch (type) {
            case 'dashboard':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <defs><filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                        <rect width="300" height="150" fill="#0a192f"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#a78bfa" stroke-width="1" class="svg-hidden draw-on-hover"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#6366f1" stroke-width="2" style="filter:url(#glow);" class="svg-hidden draw-on-hover"/>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">Sales Forecasting</text>
                    </svg>`;
                break;
            case 'trading':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#161a1d"/>
                        <path d="M30 110 L 80 40 L 130 80 L 180 60 L 230 90 L 280 30" stroke="#4f46e5" stroke-width="2" class="draw-on-hover"/>
                        <g class="svg-hidden">
                            <circle cx="80" cy="40" r="4" fill="#39ff14"/>
                            <circle cx="280" cy="30" r="4" fill="#dc143c"/>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">LSTM Analysis</text>
                    </svg>`;
                break;
            case 'toolkit':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#111827"/>
                        <defs><filter id="toolkit-glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                        <g class="toolkit-glow">
                            <path d="M150 50 L 120 67 L 120 101 L 150 118 L 180 101 L 180 67 Z" fill="rgba(139, 92, 246, 0.1)" stroke="#8b5cf6" stroke-width="1.5"/>
                            <text x="138" y="90" font-family="monospace" font-size="14" fill="#c4b5fd">API</text>
                        </g>
                        <g class="svg-hidden">
                            <path d="M110 67 L 80 84" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M190 67 L 220 84" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M110 101 L 80 118" stroke="#a78bfa" stroke-width="1"/>
                            <path d="M190 101 L 220 118" stroke="#a78bfa" stroke-width="1"/>
                            <text x="50" y="88" font-size="10" fill="white">Sentiment</text>
                            <text x="225" y="88" font-size="10" fill="white">Summarize</text>
                            <text x="50" y="122" font-size="10" fill="white">Caption</text>
                            <text x="225" y="122" font-size="10" fill="white">Generate</text>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">AI Services Toolkit</text>
                    </svg>`;
                break;
            case 'maintenance':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#2d2d2d"/>
                        <g class="pm-gear" style="transform-origin: 80px 75px;">
                            <circle cx="80" cy="75" r="30" fill="none" stroke="#add8e6" stroke-width="4" stroke-dasharray="5 5"/>
                        </g>
                        <path d="M110 75 H 280" stroke="rgba(255,140,0,0.3)" stroke-width="2"/>
                        <path d="M110 75 H 280" stroke="#ff8c00" stroke-width="2" class="draw-on-hover"/>
                        <g class="svg-hidden">
                            <rect x="230" y="65" width="50" height="20" fill="#0a0a0a" stroke="#ff8c00"/>
                            <text x="238" y="79" font-size="10" fill="#ff8c00">RUL: OK</text>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">Predictive Maintenance</text>
                    </svg>`;
                break;
            case 'churn':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="http://www.w3.org/2000/svg" class="bg-gray-900">
                        <rect width="300" height="150" fill="#003638"/>
                        <g class="churn-dot-imbalanced">
                            ${[...Array(6)].map((_, i) => `<circle cx="${40 + i * 40}" cy="50" r="5" fill="#008080" class="churn-dot"/>`).join('')}
                            <circle cx="120" cy="100" r="5" fill="#ffbf00" class="churn-dot"/>
                        </g>
                        <g class="svg-hidden">
                            <circle cx="100" cy="95" r="5" fill="#ffbf00" class="churn-dot churn-dot-synthetic"/>
                            <circle cx="140" cy="105" r="5" fill="#ffbf00" class="churn-dot churn-dot-synthetic"/>
                        </g>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">SMOTE Data Balancing</text>
                    </svg>`;
                break;
            default:
                svgContent = `<div class="w-full h-full bg-gray-800 flex items-center justify-center"><p class="text-slate-400">Project Image Placeholder</p></div>`;
        }
        return `<div class="interactive-cover-container">${svgContent}</div>`;
    };

    // Populate portfolio sections
    const projectsGrid = document.getElementById('projects-grid');
    projectsData.forEach((project, index) => {
        const card = document.createElement('div');
        card.className = "project-card card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer";
        card.innerHTML = `
            <div class="interactive-cover-container">
                <img src="${project.media[0].url}" alt="${project.title} Cover" class="w-full h-full object-cover object-center" onerror="this.onerror=null;this.src='https://placehold.co/300x150/1e1b4b/c4b5fd?text=Image+Error';">
            </div>
            <div class="p-6 flex flex-col flex-grow">
                <div>
                    <h3 class="text-xl font-bold text-white mb-2">${project.title}</h3>
                    <p class="text-slate-400 mb-4 text-sm">${project.description}</p>
                </div>
                <div class="flex flex-wrap gap-2 mt-auto pt-4">
                    ${project.skills.slice(0, 3).map(skill => `<span class="tag rounded-md px-2 py-1 text-xs">${skill}</span>`).join('')}
                </div>
            </div>`;
        card.addEventListener('click', () => openModal(index));
        projectsGrid.appendChild(card);
    });
    const skillsGrid = document.querySelector('#skills .grid');
    skillsData.forEach(category => {
        const card = document.createElement('div');
        card.className = "card-bg p-6 rounded-2xl";
        card.innerHTML = `<h3 class="text-xl font-bold text-white mb-4">${category.title}</h3><div class="flex flex-wrap gap-2">${category.skills.map(skill => `<span class="tag rounded-md px-3 py-1 text-sm">${skill}</span>`).join('')}</div>`;
        skillsGrid.appendChild(card);
    });

    // Function to render Playground apps
    const renderPlaygroundApps = () => {
        const playgroundAppsGrid = document.getElementById('playground-apps-grid');
        playgroundAppsGrid.innerHTML = ''; // Clear existing content
        playgroundAppsData.forEach(app => {
            const appCard = document.createElement('div');
            appCard.className = "card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer";
            appCard.innerHTML = `
                <img src="${app.image}" alt="${app.title}" class="w-full h-48 object-cover" onerror="this.onerror=null;this.src='https://placehold.co/600x400/1e1b4b/c4b5fd?text=App+Image';">
                <div class="p-6 flex flex-col flex-grow">
                    <div>
                        <h3 class="text-xl font-bold text-white mb-2">${app.title}</h3>
                        <p class="text-slate-400 mb-4 text-sm">${app.description}</p>
                    </div>
                    <div class="mt-auto pt-4">
                        <a href="${app.url}" target="_blank" class="inline-block bg-indigo-500 hover:bg-indigo-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300">
                            Launch App
                            <svg xmlns="http://www.w3.org/2000/svg" class="inline-block ml-1 -mt-0.5" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </a>
                    </div>
                </div>
            `;
            playgroundAppsGrid.appendChild(appCard);
        });
    };


    // Mobile menu
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    mobileMenuButton.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));

    // Typing animation
    const roles = ["AI Full Stack Developer", "Machine Learning Engineer", "Cloud & AI Engineer", "Python Backend Developer"];
    let roleIndex = 0, charIndex = 0;
    const roleTextElement = document.getElementById('role-text');
    function typeRole() { if (!roleTextElement) return; if (charIndex < roles[roleIndex].length) { roleTextElement.textContent += roles[roleIndex].charAt(charIndex++); setTimeout(typeRole, 100); } else { setTimeout(eraseRole, 2000); } }
    function eraseRole() { if (!roleTextElement) return; if (charIndex > 0) { roleTextElement.textContent = roles[roleIndex].substring(0, --charIndex); setTimeout(eraseRole, 50); } else { roleIndex = (roleIndex + 1) % roles.length; setTimeout(typeRole, 500); } }
    typeRole();

    // Modal logic
    const modal = document.getElementById('project-modal');
    const modalContent = modal.querySelector('.modal-content');
    window.openModal = (projectIndex) => {
        const project = projectsData[projectIndex];
        let galleryHtml = `<div id="media-viewer" class="mb-4 rounded-lg overflow-hidden bg-black"></div><div id="thumbnail-strip" class="flex gap-2 justify-center flex-wrap"></div>`;
        // This regex removes the [span_...](...) markers from the details before rendering
        modalContent.innerHTML = `<button class="absolute top-4 right-6 text-slate-400 hover:text-white text-3xl z-10" onclick="closeModal()">&times;</button>${galleryHtml}<div class="px-1 mt-6"><h2 class="text-3xl font-bold text-white mb-2">${project.title}</h2><p class="text-indigo-300 mb-6">${project.description}</p><h4 class="text-lg font-semibold text-white mb-2">Key Achievements:</h4><ul class="list-none space-y-2 mb-6">${project.details.map(detail => `<li class="flex items-start text-slate-300"><span class="text-indigo-400 mr-3 mt-1">▪</span><div class="flex-1">${detail.replace(/\[span_\d+\]\((start_span|end_span)\)/g, '')}</div></li>`).join('')}</ul><h4 class="text-lg font-semibold text-white mb-3">Technologies Used:</h4><div class="flex flex-wrap gap-2">${project.skills.map(skill => `<span class="tag rounded-md px-3 py-1 text-sm">${skill}</span>`).join('')}</div></div>`;
        populateGallery(projectIndex);
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    };
    window.closeModal = () => {
        modal.style.animation = 'fadeOut 0.3s ease-out forwards';
        setTimeout(() => {
            modal.style.display = 'none';
            modal.style.animation = 'fadeIn 0.3s ease-out';
            document.body.style.overflow = 'auto';
            modalContent.innerHTML = '';
        }, 300);
    };
    window.populateGallery = (projectIndex) => {
        const project = projectsData[projectIndex];
        const thumbnailStrip = document.getElementById('thumbnail-strip');
        if (project.media.length > 1) {
            thumbnailStrip.innerHTML = project.media.map((mediaItem, index) => `<div class="relative"><img src="${mediaItem.url}" class="thumbnail rounded-md w-24 h-16 object-cover" data-media-index="${index}" data-project-index="${projectIndex}"></div>`).join('');
            thumbnailStrip.querySelectorAll('.thumbnail').forEach(thumb => {
                thumb.addEventListener('click', (e) => switchMedia(e.target.dataset.mediaIndex, e.target.dataset.projectIndex));
            });
        } else { thumbnailStrip.innerHTML = ''; } // Hide thumbnail strip if only one image
        switchMedia(0, projectIndex);
    };
    window.switchMedia = (mediaIndex, projectIndex) => {
        const mediaItem = projectsData[projectIndex].media[mediaIndex];
        const viewer = document.getElementById('media-viewer');
        viewer.innerHTML = `<img src="${mediaItem.url}" alt="Project media" class="w-full h-auto max-h-[50vh] object-contain" onerror="this.onerror=null;this.src='https://placehold.co/1080x720/1e1b4b/c4b5fd?text=Error+Loading+Image';">`;
        document.querySelectorAll('#thumbnail-strip .thumbnail').forEach((thumb, i) => thumb.classList.toggle('active', i == mediaIndex));
    };
    window.onclick = (event) => { if (event.target == modal) closeModal(); };

    renderPlaygroundApps(); // Render playground apps on load

    // Q&A and Poll interactions
    const pollOptions = document.querySelectorAll('.poll-option');
    pollOptions.forEach(option => { option.addEventListener('click', () => { pollOptions.forEach(opt => { opt.disabled = true; opt.classList.add('opacity-50'); }); option.classList.add('bg-indigo-500'); document.getElementById('poll-feedback').classList.remove('hidden'); }); });

    // Background canvas animation
    const bgCanvas = document.getElementById('bg-canvas');
    if (bgCanvas) {
        const ctx = bgCanvas.getContext('2d');
        let particlesArray;
        const mouse = { x: null, y: null, radius: 0 };
        const resizeHandler = () => { bgCanvas.width = window.innerWidth; bgCanvas.height = window.innerHeight; mouse.radius = (bgCanvas.height / 120) * (bgCanvas.width / 120); initParticles(); };
        window.addEventListener('resize', resizeHandler);
        window.addEventListener('mousemove', (e) => { mouse.x = e.x; mouse.y = e.y; });
        class Particle { constructor(x, y, dx, dy) { this.x = x; this.y = y; this.directionX = dx; this.directionY = dy; this.size = (Math.random() * 2) + 1; } draw() { ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.fillStyle = 'rgba(139, 92, 246, 0.2)'; ctx.fill(); } update() { if (this.x > bgCanvas.width || this.x < 0) this.directionX = -this.directionX; if (this.y > bgCanvas.height || this.y < 0) this.directionY = -this.directionY; if (mouse.x !== null) { let dx = mouse.x - this.x; let dy = mouse.y - this.y; if (Math.hypot(dx, dy) < mouse.radius + this.size) { if (mouse.x < this.x && this.x < bgCanvas.width - this.size * 10) this.x += 5; if (mouse.x > this.x && this.x > this.size * 10) this.x -= 5; if (mouse.y < this.y && this.y < bgCanvas.height - this.size * 10) this.y += 5; if (mouse.y > this.y && this.y > this.size * 10) this.y -= 5; } } this.x += this.directionX; this.y += this.directionY; this.draw(); } }
        function initParticles() { particlesArray = []; let num = (bgCanvas.height * bgCanvas.width) / 9000; for (let i = 0; i < num; i++) { let x = Math.random() * innerWidth; let y = Math.random() * innerHeight; let dx = (Math.random() * .4) - 0.2; let dy = (Math.random() * .4) - 0.2; particlesArray.push(new Particle(x, y, dx, dy)); } }
        function animateParticles() { requestAnimationFrame(animateParticles); ctx.clearRect(0, 0, innerWidth, innerHeight); particlesArray.forEach(p => p.update()); connectParticles(); }
        function connectParticles() { let opacityValue = 1; for (let a = 0; a < particlesArray.length; a++) { for (let b = a; b < particlesArray.length; b++) { let distance = Math.hypot(particlesArray[a].x - particlesArray[b].x, particlesArray[a].y - particlesArray[b].y); if (distance < 120) { opacityValue = 1 - (distance / 120); ctx.strokeStyle = `rgba(167, 139, 250, ${opacityValue * 0.3})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(particlesArray[a].x, particlesArray[a].y); ctx.lineTo(particlesArray[b].x, particlesArray[b].y); ctx.stroke(); } } } }
        resizeHandler();
        animateParticles();
    }
});
