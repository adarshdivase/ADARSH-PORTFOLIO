// This line added to force Vercel cache refresh and demonstrate live update. Remove after verification if desired.
// --- DATA ---
const projectsData = [
    { 
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System", 
        interactive_cover: { type: 'dashboard' },
        media: [ 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce1.jpeg)' }, 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg)' } 
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
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg)' }, 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg)' } 
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
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg)' }, 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg)' } 
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
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid1.jpeg)' }, 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg)' } 
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
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg)' }, 
            { type: 'image', url: '[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg)' } 
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
        url: "[https://futureml02-jg9jkjmv3xahqnceylr8eu.streamlit.app/](https://futureml02-jg9jkjmv3xahqnceylr8eu.streamlit.app/)",
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn2.jpeg)" 
    },
    {
        title: "AI-Fueled E-commerce Analytics & Sales Forecasting System", 
        description: "A live Streamlit dashboard providing comprehensive e-commerce analytics, including sales trends, customer segmentation (RFM), and interactive visualizations for business insights.",
        url: "[https://futureml01-j2lihzs8qwk6ombkpeczut.streamlit.app/](https://futureml01-j2lihzs8qwk6ombkpeczut.streamlit.app/)",
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/ecommerce2.jpeg)" 
    },
    {
        title: "AI Services Toolkit Pro", 
        description: "Explore a suite of multi-modal AI models, including sentiment analysis, summarization, and image captioning, all served through a user-friendly interface.", 
        url: "[https://ai-toolkit-nj89aumds7l6rpjas7486m.streamlit.app/](https://ai-toolkit-nj89aumds7l6rpjas7486m.streamlit.app/)", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit1.jpeg)" 
    },
    {
        title: "Hybrid Predictive Maintenance Dashboard", 
        description: "An interactive Streamlit dashboard showcasing real-time predictive maintenance insights, including anomaly detection and remaining useful life (RUL) predictions for industrial equipment.", 
        url: "[https://smart-predictive-maintenance-en77oylapplyfegbhuzf3fy.streamlit.app/Live_Dashboard](https://smart-predictive-maintenance-en77oylapplyfegbhuzf3fy.streamlit.app/Live_Dashboard)",
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid2.jpeg)" 
    },
    {
        title: "AI-Powered Live Trading System",
        description: "An interactive Streamlit application demonstrating a deep learning-based trading system with real-time market data, technical indicators, and robust risk management.", 
        url: "[https://smart-predictive-maintenance-am3fyapk9yqcujd87tjcux.streamlit.app/](https://smart-predictive-maintenance-am3fyapk9yqcujd87tjcux.streamlit.app/)", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks2.jpeg)" 
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
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg)", 
        tags: ["Career", "Reflection", "MLOps", "Generative AI"], 
        content: `<p>Welcome to my corner of the web! I'm Adarsh Divase, an AI and Data Science enthusiast currently pursuing my B.Tech at A. C. Patil College of Engineering. My passion lies not just in understanding data, but in transforming it into intelligent, actionable solutions that can drive real-world impact. This blog is a space for me to share my journey, the projects that excite me, and the lessons I've learned along the way.</p><p>My academic and professional path has been geared towards mastering the full spectrum of the machine learning lifecycle. From constructing resilient data pipelines with <strong>Hadoop and Spark</strong> to deploying scalable microservices with <strong>Python and FastAPI</strong>, I thrive on building robust, end-to-end systems. My internship as a Python Backend Developer was a fantastic playground, allowing me to sharpen my skills in API development, database management with PostgreSQL, and Docker-based deployments, ultimately leading to significant improvements in performance and efficiency.</p><blockquote class="border-l-4 border-indigo-500 pl-4 text-slate-300 italic">"The goal is to turn data into information, and information into insight." - Carly Fiorina. This quote perfectly encapsulates my philosophy.</blockquote><p>Beyond the backend, my core interest is in the models themselves. I've delved into everything from classic predictive analytics, like the customer churn models I built using Random Forests, to the cutting edge of deep learning with <strong>CNNs, LSTMs, and Transformers</strong>. A project I'm particularly proud of is the **AI Services Toolkit Pro (Multi-Modal AI Assistant)**, where I integrated OpenAI's Whisper for transcription and YOLOv5 for object detection. It was a thrilling challenge to merge these technologies into a single, cohesive application.</p><p>My work extends to ensuring these intelligent systems are not just built but are also robust, scalable, and maintainable. This involves a strong focus on **MLOps** principles, including CI/CD pipelines, model versioning, and continuous monitoring. My experience with Docker and platforms like Render and Heroku has been crucial in deploying these complex systems efficiently. I believe that a well-engineered deployment strategy is just as important as the model itself in delivering true business value.</p><p>I'm particularly excited about the advancements in **Generative AI and Large Language Models (LLMs)</strong>, and I've been actively exploring their applications, as demonstrated by my certification in this area. The ability to create new content, synthesize information, and even generate code opens up incredible possibilities for future AI solutions.</p>`
    },
    { 
        title: "Deep Dive: Handling Imbalance in Churn Prediction", 
        date: "2025-06-20", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/churn1.jpeg)", 
        tags: ["Deep Dive", "Machine Learning", "Classification", "Data Preprocessing"], 
        content: `<p>One of the most common challenges in classification problems, like customer churn prediction, is dealing with imbalanced datasets. Often, the number of customers who churn (the minority class) is far smaller than those who don't. If left unaddressed, a model can achieve high accuracy simply by predicting the majority class every time, making it useless in practice.</p><p>Here's a conceptual Python snippet of how SMOTE can be integrated into a scikit-learn pipeline:</p><pre><code class="language-python">from imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\n# X, y are your features and labels\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Define the model and the SMOTE preprocessor\nmodel = RandomForestClassifier()\nsmote = SMOTE(random_state=42)\n\n# Create a pipeline to apply SMOTE only to the training data\npipeline = Pipeline([('smote', smote), ('classifier', model)])\n\npipeline.fit(X_train, y_train)\n\n# The pipeline handles the resampling internally during fit!\nprint(f"Model Recall: {pipeline.score(X_test, y_test)}")</code></pre><p>By applying SMOTE, I was able to increase the model's recall for the churn class by **25%**. This meant the model became significantly better at its primary job: identifying customers who are actually at risk of leaving. It's a powerful reminder that headline accuracy isn't everything; understanding and addressing the nuances of your data is what leads to truly effective models. This approach was crucial in achieving an F1-score of 0.87 and identifying 75% of churners within a 30-day window, providing actionable insights for retention strategies.</p>`
    },
    { 
        title: "The Power of Real-time Data in Trading", 
        date: "2025-06-15", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/stocks1.jpeg)", 
        tags: ["Finance", "Real-time Systems", "Data Engineering", "Trading"], 
        content: `<p>In the fast-paced world of financial trading, real-time data is not just an advantage; it's a necessity. My **AI-Powered Trading System with Risk Analytics** project heavily relied on this principle. We built robust data ingestion pipelines using technologies like **Apache Kafka** to process over 10,000 data points per minute.</p><p>This capability was critical for several reasons:</p><ul><li><strong>Timely Decision Making:</strong> Real-time data feeds directly into our LSTM models, allowing for up-to-the-minute market analysis and prediction, boosting decision efficiency by 30%.</li><li><strong>Risk Management:</strong> Integrating live data with risk metrics like VaR and Sortino Ratio ensures that our system can react swiftly to market volatility, helping to lower simulated investment risk by 22%.</li><li><strong>Competitive Edge:</strong> The ability to process and act on data faster than competitors can lead to significant gains in simulated portfolio returns.</li></ul><p>The engineering challenge lay in ensuring low-latency data flow and high throughput. By leveraging distributed streaming platforms, we created a resilient and efficient backbone for our predictive models, proving that robust data engineering is foundational to successful AI applications in finance. Our integration with the **Alpaca API** further solidified this by providing seamless access to live and historical stock data, enabling real-time bar updates and efficient order submission.</p>`
    },
    { 
        title: "Interpretable AI: Beyond the Black Box", 
        date: "2025-06-08", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid1.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/hybrid1.jpeg)", 
        tags: ["AI Ethics", "Explainable AI", "Machine Learning", "Predictive Maintenance"], 
        content: `<p>As AI models become increasingly complex, especially in critical applications like predictive maintenance, understanding *why* a model makes a certain prediction is as important as the prediction itself. This is where Explainable AI (XAI) comes into play. In my **Hybrid Predictive Maintenance System**, I focused on making the hybrid Deep Learning and Reinforcement Learning model transparent using **SHAP and LIME**.</p><p>By integrating these tools, we were able to:</p><ul><li><strong>Identify Critical Factors:</strong> Pinpoint the top 5 factors (e.g., specific sensor readings, operating conditions) that most influenced a prediction of equipment failure.</li><li><strong>Build Trust:</strong> Provide maintenance teams with clear reasons behind predicted failures, increasing their trust in the AI system.</li><li><strong>Improve Models:</strong> Insights from SHAP and LIME helped in refining features and model architecture, contributing to the 30% improvement in prediction accuracy.</li></ul><p>This commitment to interpretability ensures that our AI solutions are not just powerful but also actionable and trustworthy, bridging the gap between complex algorithms and real-world operational decisions. In the context of the Hybrid Predictive Maintenance System, this was crucial for logging simulation reports and providing actionable insights that led to an estimated 20% reduction in operational downtime costs.</p>`
    },
    { 
        title: "Unlocking Multi-Modal AI: The Toolkit Approach", 
        date: "2025-06-01", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/toolkit.jpeg)", 
        tags: ["AI", "Multi-modal AI", "Transformers", "FastAPI", "Streamlit"], 
        content: `<p>Building and deploying state-of-the-art AI models can be complex, especially when dealing with diverse data types like text, speech, and images. My **AI Services Toolkit Pro (Multi-Modal AI Assistant)** project was designed to tackle this challenge head-on by creating a unified platform for various Transformer-based AI capabilities.</p><p>Key aspects of the architecture include:</p><ul><li>A robust **FastAPI backend** handling asynchronous operations and leveraging **Pydantic models** for data validation, exposed via clean ` / api` endpoints.</li><li>An intuitive **Streamlit frontend** that provides an interactive user interface for making API calls, viewing historical interactions, and monitoring system status in real-time.</li><li>Advanced features like **Text-to-Speech (TTS)** with dynamic speaker embeddings and Speech-to-Text (STT) with automatic audio resampling, significantly enhancing accessibility and user engagement.</li></ul><p>This project demonstrates a full end-to-end MLOps pipeline, from model integration and API development to frontend deployment on platforms like Hugging Face Spaces. It highlights the power of combining specialized AI models into a user-friendly toolkit, making advanced AI capabilities accessible and actionable.</p>`
    },
    { 
        title: "Mastering MLOps: From Code to Scalable Production AI", 
        date: "2025-05-28", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/blog-mlops.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/blog-mlops.jpeg)", 
        tags: ["MLOps", "Deployment", "Scalability", "DevOps"], 
        content: `<p>In the world of AI, building a powerful model is only half the battle. The true challenge lies in deploying and maintaining these models in production reliably and at scale. This is where **MLOps (Machine Learning Operations)** becomes indispensable. My experience as a Python Backend Developer Intern, and my work on various AI projects, has deeply ingrained the importance of robust MLOps practices.</p><p>MLOps is the engineering discipline that unifies ML system development (Dev) and ML system operation (Ops). It encompasses a range of practices aimed at streamlining the machine learning lifecycle, from data collection and model training to deployment, monitoring, and continuous improvement. Key components include:</p><ul><li>**Continuous Integration/Continuous Delivery (CI/CD):** Automating the process of building, testing, and deploying models ensures rapid iteration and reduces manual errors. My work orchestrating Docker-containerized deployments to **Render and Heroku** significantly reduced deployment cycles by 40%.</li><li>**Model Versioning:** Tracking different versions of models and their associated data allows for reproducibility and rollbacks, crucial for debugging and auditing.</li><li>**Monitoring:** Continuously observing model performance, data drift, and system health in production to identify and address issues promptly.</li><li>**Scalable Infrastructure:** Designing systems that can handle increasing load. Building scalable microservices with **FastAPI and Flask** that improved API response times by 25% and handled over 10,000 daily requests with 99.9% uptime is a testament to this.</li></ul><p>Platforms like **Hugging Face Spaces** further democratize MLOps, providing easy ways to deploy and share models. Mastering these principles ensures that AI solutions are not just innovative prototypes but reliable, high-performing assets that deliver continuous business value.</p>`
    },
    { 
        title: "Data Engineering for AI: Building Resilient Data Pipelines", 
        date: "2025-05-20", 
        image: "[https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/blog-data-engineering.jpeg](https://raw.githubusercontent.com/adarshdivase/ADARSH-PORTFOLIO/main/images/blog-data-engineering.jpeg)", 
        tags: ["Data Engineering", "Big Data", "Apache Kafka", "Data Pipelines"], 
        content: `<p>At the heart of every effective AI or Machine Learning system lies a robust and reliable data pipeline. Without clean, accessible, and continuously flowing data, even the most sophisticated models are rendered useless. My academic background and project experience have provided me with profound expertise in building **resilient data engineering** solutions.</p><p>Key technologies that are fundamental to modern AI data pipelines include:</p><ul><li>**Apache Hadoop:** For distributed storage and processing of massive datasets, allowing for scalable batch processing.</li><li>**Apache Spark:** A powerful unified analytics engine for large-scale data processing, offering speed and flexibility for tasks like ETL (Extract, Transform, Load).</li><li>**Apache Kafka:** A distributed streaming platform crucial for building real-time data pipelines. In my **AI-Powered Trading System with Risk Analytics**, I established real-time data ingestion pipelines using Apache Kafka to process over 10,000 data points per minute, significantly boosting decision efficiency.</li><li>**SQL and NoSQL Databases:** Proficiently managing databases like **PostgreSQL, MySQL, and MongoDB** for structured and unstructured data storage. My work with **SQLAlchemy ORM** in managing PostgreSQL databases reduced data retrieval time by 15%.</li></ul><p>Building these resilient data pipelines ensures that AI models receive the high-quality, timely data they need to perform effectively, transforming raw data into actionable insights and driving real business impact. This foundational layer is often unseen but is paramount to the success of any AI initiative.</p>`
    }
].sort((a, b) => new Date(b.date) - new Date(a.date)); // Sort by date descending for recent posts

// --- UI LOGIC ---
document.addEventListener('DOMContentLoaded', () => {
    // Function to create interactive covers
    const createInteractiveCover = (project) => {
        const { type } = project.interactive_cover;
        let svgContent = '';
        switch (type) {
            case 'dashboard':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="bg-gray-900">
                        <defs><filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
                        <rect width="300" height="150" fill="#0a192f"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#a78bfa" stroke-width="1" class="svg-hidden draw-on-hover"/>
                        <path d="M20,130 C50,20 80,110 140,80 S200,40 280,60" stroke="#6366f1" stroke-width="2" style="filter:url(#glow);" class="svg-hidden draw-on-hover"/>
                        <text x="20" y="25" font-family="Inter, sans-serif" font-size="12" fill="#e2e8f0" class="font-bold">Sales Forecasting</text>
                    </svg>`;
                break;
            case 'trading':
                svgContent = `
                    <svg viewBox="0 0 300 150" fill="none" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="bg-gray-900">
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
                    <svg viewBox="0 0 300 150" fill="none" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="bg-gray-900">
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
                    <svg viewBox="0 0 300 150" fill="none" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="bg-gray-900">
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
                    <svg viewBox="0 0 300 150" fill="none" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="bg-gray-900">
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
                svgContent = `<div class="w-full h-full bg-gray-800 flex items-center justify-center"><p class="text-slate-400">Project Image</p></div>`;
        }
        return `<div class="interactive-cover-container">${svgContent}</div>`;
    };

    // Populate portfolio sections
    const projectsGrid = document.getElementById('projects-grid');
    projectsData.forEach((project, index) => {
        const card = document.createElement('div');
        card.className = "project-card card-bg rounded-2xl flex flex-col overflow-hidden transform hover:-translate-y-2 transition-transform duration-300 cursor-pointer";
        card.innerHTML = `
            ${createInteractiveCover(project)}
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
                <img src="${app.image}" alt="${app.title}" class="w-full h-48 object-cover" onerror="this.onerror=null;this.src='[https://placehold.co/600x400/1e1b4b/c4b5fd?text=App+Image](https://placehold.co/600x400/1e1b4b/c4b5fd?text=App+Image)';">
                <div class="p-6 flex flex-col flex-grow">
                    <div>
                        <h3 class="text-xl font-bold text-white mb-2">${app.title}</h3>
                        <p class="text-slate-400 mb-4 text-sm">${app.description}</p>
                    </div>
                    <div class="mt-auto pt-4">
                        <a href="${app.url}" target="_blank" class="inline-block bg-indigo-500 hover:bg-indigo-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300">
                            Launch App
                            <svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" class="inline-block ml-1 -mt-0.5" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
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
    const roles = ["Data Scientist", "Machine Learning Engineer", "AI Engineer"];
    let roleIndex = 0, charIndex = 0;
    const roleTextElement = document.getElementById('role-text');
    function typeRole() { if(!roleTextElement) return; if (charIndex < roles[roleIndex].length) { roleTextElement.textContent += roles[roleIndex].charAt(charIndex++); setTimeout(typeRole, 100); } else { setTimeout(eraseRole, 2000); } }
    function eraseRole() { if(!roleTextElement) return; if (charIndex > 0) { roleTextElement.textContent = roles[roleIndex].substring(0, --charIndex); setTimeout(eraseRole, 50); } else { roleIndex = (roleIndex + 1) % roles.length; setTimeout(typeRole, 500); } }
    typeRole();

    // Modal logic
    const modal = document.getElementById('project-modal');
    const modalContent = modal.querySelector('.modal-content');
    window.openModal = (projectIndex) => {
        const project = projectsData[projectIndex];
        let galleryHtml = `<div id="media-viewer" class="mb-4 rounded-lg overflow-hidden bg-black"></div><div id="thumbnail-strip" class="flex gap-2 justify-center flex-wrap"></div>`;
        modalContent.innerHTML = `<button class="absolute top-4 right-6 text-slate-400 hover:text-white text-3xl z-10" onclick="closeModal()">&times;</button>${galleryHtml}<div class="px-1 mt-6"><h2 class="text-3xl font-bold text-white mb-2">${project.title}</h2><p class="text-indigo-300 mb-6">${project.description}</p><h4 class="text-lg font-semibold text-white mb-2">Key Achievements:</h4><ul class="list-none space-y-2 mb-6">${project.details.map(detail => `<li class="flex items-start text-slate-300"><span class="text-indigo-400 mr-3 mt-1">▪</span>${detail}</li>`).join('')}</ul><h4 class="text-lg font-semibold text-white mb-3">Technologies Used:</h4><div class="flex flex-wrap gap-2">${project.skills.map(skill => `<span class="tag rounded-md px-3 py-1 text-sm">${skill}</span>`).join('')}</div></div>`;
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
        } else { thumbnailStrip.innerHTML = ''; }
        switchMedia(0, projectIndex);
    };
    window.switchMedia = (mediaIndex, projectIndex) => {
        const mediaItem = projectsData[projectIndex].media[mediaIndex];
        const viewer = document.getElementById('media-viewer');
        viewer.innerHTML = `<img src="${mediaItem.url}" alt="Project media" class="w-full h-auto max-h-[50vh] object-contain" onerror="this.onerror=null;this.src='https://placehold.co/1080x720/1e1b4b/c4b5fd?text=Error+Loading+Image';">`;
        document.querySelectorAll('#thumbnail-strip .thumbnail').forEach((thumb, i) => thumb.classList.toggle('active', i == mediaIndex));
    };
    window.onclick = (event) => { if (event.target == modal) closeModal(); };

    // Blog interactions
    const blogPostsContainer = document.getElementById('blog-posts-container');
    const blogFiltersContainer = document.getElementById('blog-filters');
    const recentPostsContainer = document.getElementById('recent-posts-container');

    const renderBlogPosts = (filter = 'All') => {
        blogPostsContainer.innerHTML = '';
        const filteredPosts = filter === 'All' ? blogPostsData : blogPostsData.filter(p => p.tags.includes(filter));
        filteredPosts.forEach((post, index) => {
            const article = document.createElement('article');
            article.className = 'card-bg p-8 rounded-2xl';
            article.innerHTML = `
                <img src="${post.image}" alt="${post.title}" class="rounded-lg mb-6 w-full h-64 object-cover">
                <div class="flex flex-wrap gap-2 mb-4">
                    ${post.tags.map(tag => `<span class="tag rounded-md px-2 py-1 text-xs">${tag}</span>`).join('')}
                </div>
                <h3 class="text-2xl font-bold text-white mb-2">${post.title}</h3>
                <p class="text-sm text-slate-400 mb-4">Posted on ${post.date}</p>
                <div id="blog-content-${index}" class="prose-custom text-slate-400 blog-content-truncated">
                   ${post.content}
                </div>
                <button class="read-more-btn text-indigo-400 hover:text-indigo-300 mt-4 text-sm font-semibold" data-post-index="${index}">Read More</button>
            `;
            blogPostsContainer.appendChild(article);
        });
        hljs.highlightAll(); // Highlight code snippets
        
        // Add event listeners for "Read More" buttons
        document.querySelectorAll('.read-more-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const postIndex = e.target.dataset.postIndex;
                const contentDiv = document.getElementById(`blog-content-${postIndex}`);
                contentDiv.classList.toggle('blog-content-truncated');
                contentDiv.classList.toggle('blog-content-expanded');
                e.target.textContent = contentDiv.classList.contains('blog-content-truncated') ? 'Read More' : 'Read Less';
            });
        });
    };
    
    const renderBlogFilters = () => {
        const allTags = ['All', ...new Set(blogPostsData.flatMap(p => p.tags))].sort(); // Sort tags alphabetically
        blogFiltersContainer.innerHTML = allTags.map(tag => 
            `<button class="blog-tag-filter tag px-4 py-2 rounded-lg ${tag === 'All' ? 'active' : ''}" data-filter="${tag}">${tag}</button>`
        ).join('');
        
        blogFiltersContainer.querySelectorAll('.blog-tag-filter').forEach(button => {
            button.addEventListener('click', (e) => {
                const filter = e.target.dataset.filter;
                blogFiltersContainer.querySelector('.active').classList.remove('active');
                e.target.classList.add('active');
                renderBlogPosts(filter);
            });
        });
    };

    const renderRecentPosts = () => {
        recentPostsContainer.innerHTML = '';
        // Get the top 3 most recent posts (already sorted by date in data)
        const recentPosts = blogPostsData.slice(0, 3); 
        recentPosts.forEach(post => {
            const postLink = document.createElement('a');
            postLink.href = "#blog"; // Link to the blog section
            postLink.className = "block card-bg p-4 rounded-lg hover:bg-slate-800 transition-colors";
            postLink.innerHTML = `
                <p class="text-sm font-semibold text-white">${post.title}</p>
                <p class="text-xs text-slate-400">${post.date}</p>
            `;
            recentPostsContainer.appendChild(postLink);
        });
    };

    renderBlogFilters();
    renderBlogPosts();
    renderRecentPosts(); // Render recent posts on load
    renderPlaygroundApps(); // Render playground apps on load

    // Q&A and Poll interactions
    const commentForm = document.getElementById('comment-form');
    commentForm.addEventListener('submit', (e) => { e.preventDefault(); e.target.reset(); /* Replaced alert with console.log for Canvas environment */ console.log("Thank you! Your question has been submitted."); });
    const pollOptions = document.querySelectorAll('.poll-option');
    pollOptions.forEach(option => { option.addEventListener('click', () => { pollOptions.forEach(opt => { opt.disabled = true; opt.classList.add('opacity-50'); }); option.classList.add('bg-indigo-500'); document.getElementById('poll-feedback').classList.remove('hidden'); }); });

    // Background canvas animation
    const bgCanvas = document.getElementById('bg-canvas');
    const ctx = bgCanvas.getContext('2d');
    let particlesArray;
    const mouse = { x: null, y: null, radius: 0 };
    const resizeHandler = () => { bgCanvas.width = window.innerWidth; bgCanvas.height = window.innerHeight; mouse.radius = (bgCanvas.height / 120) * (bgCanvas.width / 120); initParticles(); };
    window.addEventListener('resize', resizeHandler);
    window.addEventListener('mousemove', (e) => { mouse.x = e.x; mouse.y = e.y; });
    class Particle { constructor(x, y, dx, dy) { this.x = x; this.y = y; this.directionX = dx; this.directionY = dy; this.size = (Math.random() * 2) + 1; } draw() { ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.fillStyle = 'rgba(139, 92, 246, 0.2)'; ctx.fill(); } update() { if (this.x > bgCanvas.width || this.x < 0) this.directionX = -this.directionX; if (this.y > bgCanvas.height || this.y < 0) this.directionY = -this.directionY; let dx = mouse.x - this.x; let dy = mouse.y - this.y; if (Math.hypot(dx, dy) < mouse.radius + this.size) { if (mouse.x < this.x && this.x < bgCanvas.width - this.size * 10) this.x += 5; if (mouse.x > this.x && this.x > this.size * 10) this.x -= 5; if (mouse.y < this.y && this.y < bgCanvas.height - this.size * 10) this.y += 5; if (mouse.y > this.y && this.y > this.size * 10) this.y -= 5; } this.x += this.directionX; this.y += this.directionY; this.draw(); } } // FIX: Corrected 'the.size' to 'this.size' and 'the.y' to 'this.y'
    function initParticles() { particlesArray = []; let num = (bgCanvas.height * bgCanvas.width) / 9000; for (let i = 0; i < num; i++) { let x = Math.random() * innerWidth; let y = Math.random() * innerHeight; let dx = (Math.random() * .4) - 0.2; let dy = (Math.random() * .4) - 0.2; particlesArray.push(new Particle(x, y, dx, dy)); } }
    function animateParticles() { requestAnimationFrame(animateParticles); ctx.clearRect(0, 0, innerWidth, innerHeight); particlesArray.forEach(p => p.update()); connectParticles(); }
    function connectParticles() { let opacityValue = 1; for (let a = 0; a < particlesArray.length; a++) { for (let b = a; b < particlesArray.length; b++) { let distance = Math.hypot(particlesArray[a].x - particlesArray[b].x, particlesArray[a].y - particlesArray[b].y); if (distance < 120) { opacityValue = 1 - (distance / 120); ctx.strokeStyle = `rgba(167, 139, 250, ${opacityValue * 0.3})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(particlesArray[a].x, particlesArray[a].y); ctx.lineTo(particlesArray[b].x, particlesArray[b].y); ctx.stroke(); } } } }
    resizeHandler();
    animateParticles();
    console.log("Adarsh's Portfolio script loaded and executed!"); // Added for verification
});
You are providing the `script.js` content again, and it **still contains the critical typo** that causes the `tC is not defined` error:

```javascript
if (mouse.y > this.y && this.y > the.size * 10) the.y -= 5;
