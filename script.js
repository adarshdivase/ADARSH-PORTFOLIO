// This line added to force Vercel cache refresh and demonstrate live update. Remove after verification if desired.
// --- DATA ---
const projectsData = [
    { 
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System", 
        interactive_cover: { type: 'dashboard' },
        media: [ 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce1.jpeg' }, 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg' } 
        ], 
        description: "An AI-powered platform for e-commerce analytics and sales forecasting, leveraging Facebook Prophet and interactive dashboards to drive revenue strategy and reduce stockouts.", 
        details: [ 
            "Constructed a robust sales forecasting model using **Facebook Prophet**, achieving 92% MAPE for quarterly sales predictions and effectively reducing stockouts by 10%.", 
            "Engineered comprehensive data processing pipelines using **Pandas and NumPy**, preparing data for time series forecasting and business intelligence.", 
            "Created interactive and dynamic dashboards using **Plotly Express and Streamlit**, visualizing key e-commerce metrics and forecast performance, leading to a 15% increase in revenue strategy impact and a 5% uplift in overall quarterly sales revenue." 
        ],
        skills: ["Streamlit", "Prophet", "Pandas", "NumPy", "Plotly Express", "Data Engineering", "Business Intelligence", "Predictive Analytics"] 
    },
    { 
        title: "AI-Powered Trading System with Risk Analytics", 
        interactive_cover: { type: 'trading' },
        media: [ 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg' }, 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg' } 
        ], 
        description: "A real-time AI-driven algorithmic trading system deployed on Streamlit, providing live market data, technical indicators, and automated trade execution with robust risk management protocols.", 
        details: [ 
            "Engineered a real-time AI-driven algorithmic trading system deployed on **Streamlit**, providing live market data, technical indicators, and automated trade execution, leading to a 5% increase in simulated portfolio returns.", 
            "Developed an enhanced trading strategy combining **MACD, RSI, and Bollinger Bands** for multi-factor signal generation with volume confirmation.", 
            "Implemented robust risk management protocols, including dynamic position sizing based on portfolio risk (2% per trade), and maximum daily loss limits (2%), ensuring capital preservation.", 
            "Integrated with **Alpaca API** for fetching historical and live stock data, enabling real-time bar updates and seamless order submission." 
        ], 
        skills: ["Streamlit", "NumPy", "Pandas", "PyTorch", "Scikit-learn", "Plotly", "Alpaca API", "Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)", "Financial Analytics", "VaR", "Monte Carlo Simulation", "Real-Time Systems", "Quantitative Finance", "Time Series Forecasting"] 
    },
    { 
        title: "AI Services Toolkit Pro (Multi-Modal AI Assistant)", 
        interactive_cover: { type: 'toolkit' },
        media: [ 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg' }, 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg' } 
        ], 
        description: "Architected and deployed a comprehensive, integrated Multi-Modal AI Toolkit on Hugging Face Spaces, integrating 9 state-of-the-art Transformer pipelines for diverse AI capabilities.", 
        details: [ 
            "Architected and deployed a comprehensive, integrated Multi-Modal AI Toolkit on **Hugging Face Spaces**, integrating 9 state-of-the-art Transformer pipelines for diverse AI capabilities.", 
            "Developed a robust **FastAPI** backend with asynchronous operations and **Pydantic models**, exposed via `/api` endpoints.", 
            "Designed an intuitive **Streamlit** frontend for interactive AI service interaction, API call history, and system status monitoring.", 
            "Implemented Text-to-Speech (TTS) with dynamic speaker embeddings and Speech-to-Text (STT) with automatic audio resampling, enhancing accessibility by 15% for 5,000 daily users.", 
            "Utilized key libraries: FastAPI, Streamlit, Hugging Face Transformers, PyTorch, soundfile, librosa." 
        ], 
        skills: ["FastAPI", "Streamlit", "Hugging Face Transformers", "PyTorch", "soundfile", "librosa", "Docker", "Full-Stack Development", "MLOps", "Whisper API", "YOLOv5", "NLP", "Speech-to-Text", "Text-to-Speech", "Real-time Systems", "Computer Vision"] 
    },
    { 
        title: "Hybrid Predictive Maintenance System", 
        interactive_cover: { type: 'maintenance' },
        media: [ 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid1.jpeg' }, 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg' } 
        ], 
        description: "Developed and deployed an integrated Hybrid Predictive Maintenance system on Streamlit, combining supervised learning (LSTM) and reinforcement learning for optimal maintenance recommendations.", 
        details: [ 
            "Developed and deployed an integrated Hybrid Predictive Maintenance system on **Streamlit**, integrating supervised learning (**LSTM**) and reinforcement learning for optimal maintenance recommendations.", 
            "Engineered an LSTM-based deep learning model using **TensorFlow/Keras** to predict machine health and Remaining Useful Life (RUL) from synthetic time-series data, achieving a 30% improvement in prediction accuracy.", 
            "Designed a comprehensive Streamlit multi-page application with a 'Live Dashboard' for real-time monitoring and a 'Historical Explorer' for data analysis.", 
            "Established a persistent **SQLite database** to log simulation reports, including health metrics and explainability insights (simulated **SHAP/LIME**), reducing operational downtime costs by an estimated 20%, resulting in annual savings of $50,000." 
        ], 
        skills: ["Streamlit", "TensorFlow", "Keras", "NumPy", "Pandas", "SQLite3", "Deep Learning (CNNs)", "Reinforcement Learning (PPO)", "SHAP", "LIME", "Plotly", "Python", "Data Analytics", "Anomaly Detection"] 
    },
    { 
        title: "Customer Churn Prediction and API Deployment", 
        interactive_cover: { type: 'churn' },
        media: [ 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg' }, 
            { type: 'image', url: 'https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg' } 
        ], 
        description: "Architected and deployed an integrated Customer Churn Prediction system on Streamlit with a FastAPI backend for model inference, achieving high accuracy and efficient real-time predictions.", 
        details: [ 
            "Architected and deployed an integrated Customer Churn Prediction system on **Streamlit** with a **FastAPI** backend for model inference.", 
            "Developed a robust churn prediction model (F1-score of 0.87) using advanced ML techniques, identifying 75% of potential churners within 30 days.", 
            "Engineered data preprocessing pipelines with **SMOTE**, increasing model recall for the minority class by 25%.", 
            "Launched a high-performance RESTful API using **FastAPI and Uvicorn**, achieving sub-100ms inference latency and handling up to 500 requests/second." 
        ], 
        skills: ["FastAPI", "Streamlit", "Scikit-learn", "Pandas", "NumPy", "imblearn", "Random Forest", "Neural Networks", "MLOps", "Model Deployment", "REST API", "Python", "Classification"] 
    }
];

// Data for Playground section
const playgroundAppsData = [
    {
        title: "AI-Powered Customer Churn Prediction",
        description: "An interactive Streamlit application demonstrating a machine learning model that predicts customer churn, allowing users to input customer data and see real-time predictions.",
        url: "https://futureml02-jg9jkjmv3xahqnceylr8eu.streamlit.app/",
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg" 
    },
    {
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System", 
        description: "A live Streamlit dashboard providing comprehensive e-commerce analytics, including sales trends, customer segmentation (RFM), and interactive visualizations for business insights.",
        url: "https://futureml01-j2lihzs8qwk6ombkpeczut.streamlit.app/",
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg" 
    },
    {
        title: "AI Services Toolkit Pro", 
        description: "Explore a suite of multi-modal AI models, including sentiment analysis, summarization, and image captioning, all served through a user-friendly interface.", 
        url: "https://ai-toolkit-nj89aumds7l6rpjas7486m.streamlit.app/", 
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg" 
    },
    {
        title: "Hybrid Predictive Maintenance Dashboard", 
        description: "An interactive Streamlit dashboard showcasing real-time predictive maintenance insights, including anomaly detection and remaining useful life (RUL) predictions for industrial equipment.", 
        url: "https://smart-predictive-maintenance-en77oylapplyfegbhuzf3fy.streamlit.app/Live_Dashboard",
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg" 
    },
    {
        title: "AI-Powered Live Trading System",
        description: "An interactive Streamlit application demonstrating a deep learning-based trading system with real-time market data, technical indicators, and robust risk management.", 
        url: "https://smart-predictive-maintenance-am3fyapk9yqcujd87tjcux.streamlit.app/", 
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg" 
    }
];

const skillsData = [
    { title: 'Programming Languages & Frameworks', skills: ['Python (NumPy, Pandas)', 'SQL', 'FastAPI', 'Flask', 'Data Structures', 'Algorithms'] }, 
    { title: 'Machine Learning', skills: ['Scikit-learn', 'XGBoost', 'Random Forests', 'SVMs', 'Regression', 'Classification', 'Clustering', 'Feature Engineering', 'Model Evaluation', 'SMOTE', 'Reinforcement Learning'] }, 
    { title: 'Deep Learning', skills: ['TensorFlow', 'PyTorch', 'Keras', 'CNNs', 'RNNs (LSTMs, GRUs)', 'Transformers', 'Transfer Learning', 'GANs'] }, 
    { title: 'MLOps & Deployment', skills: ['Docker', 'Kubernetes (basics)', 'Render', 'Heroku', 'Git', 'Uvicorn', 'RESTful API', 'Microservices', 'CI/CD', 'Model Versioning', 'Monitoring', 'Hugging Face Spaces'] }, 
    { title: 'Big Data & Databases', skills: ['Hadoop', 'Spark', 'Apache Kafka', 'PostgreSQL', 'MySQL', 'SQLAlchemy', 'MongoDB', 'SQLite3'] }, 
    { title: 'Data Visualization & BI', skills: ['Plotly', 'Matplotlib', 'Seaborn', 'Dash', 'Tableau (conceptual)', 'Streamlit'] }, 
    { title: 'Specialized Tools', skills: ['SHAP', 'LIME', 'Prophet', 'Whisper API', 'YOLOv5', 'OpenAI API', 'LLMs (ChatGPT)', 'Financial Modeling', 'Monte Carlo Simulation', 'A/B Testing', 'Alpaca API', 'Technical Indicators (MACD, RSI, Bollinger Bands, SMAs)'] } 
];

const blogPostsData = [
     { 
        title: "Architecting Intelligence: My Journey in Tech", 
        date: "2025-06-26", 
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg", 
        tags: ["Career", "Reflection", "MLOps", "Generative AI"], 
        content: `<p>Welcome to my corner of the web! I'm Adarsh Divase, an AI and Data Science enthusiast currently pursuing my B.Tech at A. C. Patil College of Engineering. My passion lies not just in understanding data, but in transforming it into intelligent, actionable solutions that can drive real-world impact. This blog is a space for me to share my journey, the projects that excite me, and the lessons I've learned along the way.</p><p>My academic and professional path has been geared towards mastering the full spectrum of the machine learning lifecycle. From constructing resilient data pipelines with <strong>Hadoop and Spark</strong> to deploying scalable microservices with <strong>Python and FastAPI</strong>, I thrive on building robust, end-to-end systems. My internship as a Python Backend Developer was a fantastic playground, allowing me to sharpen my skills in API development, database management with PostgreSQL, and Docker-based deployments, ultimately leading to significant improvements in performance and efficiency.</p><blockquote class="border-l-4 border-indigo-500 pl-4 text-slate-300 italic">"The goal is to turn data into information, and information into insight." - Carly Fiorina. This quote perfectly encapsulates my philosophy.</blockquote><p>Beyond the backend, my core interest is in the models themselves. I've delved into everything from classic predictive analytics, like the customer churn models I built using Random Forests, to the cutting edge of deep learning with <strong>CNNs, LSTMs, and Transformers</strong>. A project I'm particularly proud of is the **AI Services Toolkit Pro (Multi-Modal AI Assistant)**, where I integrated OpenAI's Whisper for transcription and YOLOv5 for object detection. It was a thrilling challenge to merge these technologies into a single, cohesive application.</p><p>My work extends to ensuring these intelligent systems are not just built but are also robust, scalable, and maintainable. This involves a strong focus on **MLOps** principles, including CI/CD pipelines, model versioning, and continuous monitoring. My experience with Docker and platforms like Render and Heroku has been crucial in deploying these complex systems efficiently. I believe that a well-engineered deployment strategy is just as important as the model itself in delivering true business value.</p><p>I'm particularly excited about the advancements in **Generative AI and Large Language Models (LLMs)</strong>, and I've been actively exploring their applications, as demonstrated by my certification in this area. The ability to create new content, synthesize information, and even generate code opens up incredible possibilities for future AI solutions.</p>`
    },
    { 
        title: "Deep Dive: Handling Imbalance in Churn Prediction", 
        date: "2025-06-20", 
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg", 
        tags: ["Deep Dive", "Machine Learning", "Classification", "Data Preprocessing"], 
        content: `<p>One of the most common challenges in classification problems, like customer churn prediction, is dealing with imbalanced datasets. Often, the number of customers who churn (the minority class) is far smaller than those who don't. If left unaddressed, a model can achieve high accuracy simply by predicting the majority class every time, making it useless in practice.</p><p>Here's a conceptual Python snippet of how SMOTE can be integrated into a scikit-learn pipeline:</p><pre><code class="language-python">from imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\n# X, y are your features and labels\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Define the model and the SMOTE preprocessor\nmodel = RandomForestClassifier()\nsmote = SMOTE(random_state=42)\n\n# Create a pipeline to apply SMOTE only to the training data\npipeline = Pipeline([('smote', smote), ('classifier', model)])\n\npipeline.fit(X_train, y_train)\n\n# The pipeline handles the resampling internally during fit!\nprint(f"Model Recall: {pipeline.score(X_test, y_test)}")</code></pre><p>By applying SMOTE, I was able to increase the model's recall for the churn class by **25%**. This meant the model became significantly better at its primary job: identifying customers who are actually at risk of leaving. It's a powerful reminder that headline accuracy isn't everything; understanding and addressing the nuances of your data is what leads to truly effective models. This approach was crucial in achieving an F1-score of 0.87 and identifying 75% of churners within a 30-day window, providing actionable insights for retention strategies.</p>`
    },
    { 
        title: "The Power of Real-time Data in Trading", 
        date: "2025-06-15", 
        image: "https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg", 
        tags: ["Finance", "Real-time Systems", "Data Engineering", "Trading"], 
        content: `<p>In the fast-paced world of financial trading, real-time data is not just an advantage; it's a necessity. My **AI-Powered Trading System with Risk Analytics** project heavily relied on this principle. We built robust data ingestion pipelines using technologies like **Apache Kafka** to process over 10,000 data points per minute.</
