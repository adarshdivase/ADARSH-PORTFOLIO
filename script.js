// --- DATA ---
const projectsData = [
    { 
        title: "E-Com Insight Dashboard", 
        interactive_cover: { type: 'dashboard' },
        media: [ { type: 'image', url: 'https://i.imgur.com/3ZkE8aD.png' }, { type: 'image', url: 'https://i.imgur.com/9qZ0A8a.png' } ], 
        description: "An advanced analytics and sales forecasting platform for e-commerce, designed to provide deep insights into customer behavior and market trends.", 
        details: [ 
            "Engineered a real-time analytics dashboard using Plotly and Dash to visualize key performance indicators, enabling stakeholders to make data-driven decisions swiftly.", 
            "Implemented the Prophet time series forecasting model, achieving a 92% accuracy (MAPE) for quarterly sales predictions and effectively reducing stockouts by 10%.", 
            "Developed a robust and scalable multithreaded Python backend, capable of supporting over 100 concurrent users with a 40% reduction in query response time, ensuring seamless data access and analysis." 
        ],
        skills: ["Prophet", "SQL", "Python (Pandas, NumPy)", "Plotly", "Dash", "Multithreading", "Data Engineering", "Business Intelligence", "Predictive Analytics"] 
    },
    { 
        title: "AI-Powered Trading System", 
        interactive_cover: { type: 'trading' },
        media: [ { type: 'image', url: 'https://i.imgur.com/CAbpXpQ.png' }, { type: 'image', url: 'https://i.imgur.com/s4n4g5K.png' } ], 
        description: "A sophisticated deep learning system for financial market forecasting, integrated with advanced risk analytics to optimize portfolio performance and minimize exposure.", 
        details: [ 
            "Conceptualized and implemented deep learning forecasting models using LSTM networks, which improved prediction accuracy by 18% over traditional methods and led to a 5% increase in simulated portfolio returns.", 
            "Incorporated advanced financial risk metrics such as Sortino Ratio, Value at Risk (VaR), and Monte Carlo Simulation, effectively lowering simulated investment risk by 22%.", 
            "Established real-time data ingestion pipelines using Apache Kafka, processing over 10,000 data points per minute, thereby boosting decision efficiency by 30% for timely trading strategies." 
        ], 
        skills: ["Deep Learning (LSTMs, RNNs)", "Financial Analytics", "Sortino Ratio", "VaR", "Monte Carlo Simulation", "Python", "Apache Kafka", "Real-Time Systems", "Quantitative Finance", "Time Series Forecasting"] 
    },
    { 
        title: "AI Services Toolkit (Multi-Modal Assistant)", 
        interactive_cover: { type: 'toolkit' },
        media: [ { type: 'image', url: 'https://i.imgur.com/3i4b5nS.png' }, { type: 'image', url: 'https://i.imgur.com/rGfFvWq.png' } ], 
        description: "A full-stack web application serving a suite of self-hosted AI models, including a multi-modal assistant, through a clean, interactive user interface built with React and FastAPI.", 
        details: [ 
            "Engineered a multi-modal AI assistant utilizing OpenAI Whisper for high-fidelity speech-to-text transcription (95% accuracy) and YOLOv5 for real-time object detection (90% object accuracy).", 
            "Developed a robust FastAPI backend optimized for asynchronous performance, significantly lowering API latency by 35% to achieve sub-200ms response times for AI model inferences.", 
            "Deployed real-time Text-to-Speech (TTS) translation capabilities, boosting accessibility and user engagement for approximately 5,000 daily users by 15%." 
        ], 
        skills: ["React", "FastAPI (Asynchronous)", "Hugging Face Transformers", "Docker", "Full-Stack Development", "MLOps", "Whisper API", "YOLOv5", "NLP", "Speech-to-Text", "Text-to-Speech", "Real-time Systems", "Computer Vision"] 
    },
    { 
        title: "SMART Predictive Maintenance System", 
        interactive_cover: { type: 'maintenance' },
        media: [ { type: 'image', url: 'https://i.imgur.com/3ZkE8aD.png' }, { type: 'image', url: 'https://i.imgur.com/5uV2pMS.png' } ], 
        description: "A hybrid AI system combining Deep Learning and Reinforcement Learning to predict equipment failures and recommend optimal maintenance schedules, reducing operational costs.", 
        details: [ 
            "Developed a framework integrating Convolutional Neural Networks (CNNs) for anomaly detection and a PPO-based Reinforcement Learning agent for optimal maintenance scheduling, achieving a 30% improvement in prediction accuracy.", 
            "Utilized SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) for model interpretability, successfully identifying the top 5 critical factors contributing to equipment breakdowns.", 
            "Architected a real-time dashboard using Plotly, which facilitated proactive maintenance and reduced estimated operational downtime costs by 20% (estimated $50,000 annual savings)." 
        ], 
        skills: ["Deep Learning (CNNs)", "Reinforcement Learning (PPO)", "SHAP", "LIME", "Plotly", "Python", "Data Analytics", "Anomaly Detection", "TensorFlow"] 
    },
    { 
        title: "Customer Churn Prediction System", 
        interactive_cover: { type: 'churn' },
        media: [ { type: 'image', url: 'https://i.imgur.com/8fH7z9T.png' }, { type: 'image', url: 'https://i.imgur.com/pXv9z4L.png' } ], 
        description: "An end-to-end classification system designed to identify at-risk customers, featuring advanced data balancing techniques and robust API deployment for real-time predictions.", 
        details: [ 
            "Architected and trained a customer churn prediction model using Random Forests and Neural Networks, achieving an F1-score of 0.87 and successfully identifying 75% of churners within a 30-day window.", 
            "Implemented SMOTE (Synthetic Minority Over-sampling Technique) to effectively rebalance the imbalanced dataset, which significantly boosted the model's recall for the minority (churn) class by 25%.", 
            "Launched the final model as a high-performance RESTful API using FastAPI and Uvicorn, capable of handling up to 500 requests per second with sub-100ms inference latency, ensuring rapid and scalable predictions." 
        ], 
        skills: ["Random Forest", "Neural Networks", "SMOTE", "FastAPI", "Uvicorn", "Scikit-learn", "MLOps", "Model Deployment", "REST API", "Python", "Classification"] 
    }
];

// New data for Playground section
const playgroundAppsData = [
    {
        title: "AI-Powered Customer Churn Prediction",
        description: "An interactive Streamlit application demonstrating a machine learning model that predicts customer churn, allowing users to input customer data and see real-time predictions.",
        url: "https://futureml02-jg9jkjmv3xahqnceylr8eu.streamlit.app/",
        image: "https://placehold.co/600x400/4c1d95/c4b5fd?text=Churn+Prediction+App" // Placeholder image
    },
    {
        title: "E-Com Insight Dashboard",
        description: "A live Streamlit dashboard providing comprehensive e-commerce analytics, including sales trends, customer segmentation (RFM), and interactive visualizations for business insights.",
        url: "https://futureml01-j2lihzs8qwk6ombkpeczut.streamlit.app/",
        image: "https://placehold.co/600x400/1e3a8a/93c5fd?text=E-Com+Dashboard+App" // Placeholder image
    },
    {
        title: "AI Services Toolkit",
        description: "Explore a suite of AI models, including sentiment analysis, summarization, and image captioning, all served through a user-friendly interface.",
        url: "https://ai-toolkit-nj89aumds7l6rpjas7486m.streamlit.app/", 
        image: "https://placehold.co/600x400/8b5cf6/c4b5fd?text=AI+Toolkit+App" // Placeholder image
    },
    {
        title: "SMART Predictive Maintenance Dashboard (Live)",
        description: "An interactive Streamlit dashboard showcasing real-time predictive maintenance insights, including anomaly detection and remaining useful life (RUL) predictions for industrial equipment.",
        url: "https://smart-predictive-maintenance-en77oylapplyfegbhuzf3fy.streamlit.app/Live_Dashboard",
        image: "https://placehold.co/600x400/2d2d2d/add8e6?text=Predictive+Maintenance+App" // Placeholder image
    },
    {
        title: "AI-Powered Live Trading System",
        description: "An interactive Streamlit application demonstrating a deep learning-based trading system with real-time data analysis and risk management.",
        url: "https://smart-predictive-maintenance-am3fyapk9yqcujd87tjcux.streamlit.app/", 
        image: "https://placehold.co/600x400/4a0e7e/c4b5fd?text=Trading+System+App" // Placeholder image
    }
];

const skillsData = [
    { title: 'Languages & Frameworks', skills: ['Python', 'SQL', 'FastAPI', 'Flask', 'Data Structures', 'Algorithms'] },
    { title: 'Machine & Deep Learning', skills: ['Scikit-learn', 'TensorFlow', 'PyTorch', 'CNNs', 'RNNs (LSTMs)', 'Transformers', 'XGBoost', 'GANs'] },
    { title: 'MLOps & Deployment', skills: ['Docker', 'Kubernetes (basics)', 'Render', 'Heroku', 'Git', 'CI/CD', 'RESTful API'] },
    { title: 'Big Data & Databases', skills: ['Hadoop', 'Spark', 'Apache Kafka', 'PostgreSQL', 'MySQL', 'MongoDB'] },
    { title: 'Data Viz & BI', skills: ['Plotly', 'Matplotlib', 'Seaborn', 'Dash', 'Tableau (conceptual)'] },
    { title: 'Specialized Tools', skills: ['SHAP', 'LIME', 'Prophet', 'Whisper API', 'YOLOv5', 'OpenAI API', 'LLMs', 'Reinforcement Learning'] }
];

const blogPostsData = [
     { 
        title: "Architecting Intelligence: My Journey in Tech", 
        date: "2025-06-26", 
        image: "https://images.unsplash.com/photo-1599658880436-c07f3e834169?auto=format&fit=crop&w=1080&q=80", 
        tags: ["Career", "Reflection", "MLOps"], 
        content: `<p>Welcome to my corner of the web! I'm Adarsh Divase, an AI and Data Science enthusiast currently pursuing my B.Tech at A. C. Patil College of Engineering. My passion lies not just in understanding data, but in transforming it into intelligent, actionable solutions that can drive real-world impact. This blog is a space for me to share my journey, the projects that excite me, and the lessons I've learned along the way.</p><p>My academic and professional path has been geared towards mastering the full spectrum of the machine learning lifecycle. From constructing resilient data pipelines with <strong>Hadoop and Spark</strong> to deploying scalable microservices with <strong>Python and FastAPI</strong>, I thrive on building robust, end-to-end systems. My internship as a Python Backend Developer was a fantastic playground, allowing me to sharpen my skills in API development, database management with PostgreSQL, and Docker-based deployments, ultimately leading to significant improvements in performance and efficiency.</p><blockquote class="border-l-4 border-indigo-500 pl-4 text-slate-300 italic">"The goal is to turn data into information, and information into insight." - Carly Fiorina. This quote perfectly encapsulates my philosophy.</blockquote><p>Beyond the backend, my core interest is in the models themselves. I've delved into everything from classic predictive analytics, like the customer churn models I built using Random Forests, to the cutting edge of deep learning with <strong>CNNs, LSTMs, and Transformers</strong>. A project I'm particularly proud of is the AI-driven multi-modal assistant, where I integrated OpenAI's Whisper for transcription and YOLOv5 for object detection. It was a thrilling challenge to merge these technologies into a single, cohesive application.</p><p>My work extends to ensuring these intelligent systems are not just built but are also robust, scalable, and maintainable. This involves a strong focus on MLOps principles, including CI/CD pipelines, model versioning, and continuous monitoring. My experience with Docker and platforms like Render and Heroku has been crucial in deploying these complex systems efficiently. I believe that a well-engineered deployment strategy is just as important as the model itself in delivering true business value.</p>`
    },
    { 
        title: "Deep Dive: Handling Imbalance in Churn Prediction", 
        date: "2025-06-20", 
        image: "https://images.unsplash.com/photo-1591696205602-2f950c417cb9?auto=format&fit=crop&w=1080&q=80", 
        tags: ["Deep Dive", "Machine Learning", "Classification"], 
        content: `<p>One of the most common challenges in classification problems, like customer churn prediction, is dealing with imbalanced datasets. Often, the number of customers who churn (the minority class) is far smaller than those who don't. If left unaddressed, a model can achieve high accuracy simply by predicting the majority class every time, making it useless in practice.</p><p>In my Customer Churn Prediction project, I tackled this head-on. The key was using the <strong>SMOTE (Synthetic Minority Over-sampling TEchnique)</strong> algorithm. Instead of just duplicating existing minority class samples, SMOTE intelligently generates new, synthetic samples that are close to the existing ones in the feature space. This provides a richer, more balanced dataset for the model to train on.</p><p>Here's a conceptual Python snippet of how SMOTE can be integrated into a scikit-learn pipeline:</p><pre><code class="language-python">from imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\n# X, y are your features and labels\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Define the model and the SMOTE preprocessor\nmodel = RandomForestClassifier()\nsmote = SMOTE(random_state=42)\n\n# Create a pipeline to apply SMOTE only to the training data\npipeline = Pipeline([('smote', smote), ('classifier', model)])\n\npipeline.fit(X_train, y_train)\n\n# The pipeline handles the resampling internally during fit!\nprint(f"Model Recall: {pipeline.score(X_test, y_test)}")</code></pre><p>By applying SMOTE, I was able to increase the model's recall for the churn class by <strong>25%</strong>. This meant the model became significantly better at its primary job: identifying customers who are actually at risk of leaving. It's a powerful reminder that headline accuracy isn't everything; understanding and addressing the nuances of your data is what leads to truly effective models. This approach was crucial in achieving an F1-score of 0.87 and identifying 75% of churners within a 30-day window, providing actionable insights for retention strategies.</p>`
    },
    {
        title: "The Power of Real-time Data in Trading",
        date: "2025-06-15",
        image: "https://images.unsplash.com/photo-1590283120015-385078a63584?auto=format&fit=crop&w=1080&q=80",
        tags: ["Finance", "Real-time Systems", "Data Engineering"],
        content: `<p>In the fast-paced world of financial trading, real-time data is not just an advantage; it's a necessity. My AI-Powered Trading System project heavily relied on this principle. We built robust data ingestion pipelines using technologies like <strong>Apache Kafka</strong> to process over 10,000 data points per minute.</p><p>This capability was critical for several reasons:</p><ul><li><strong>Timely Decision Making:</strong> Real-time data feeds directly into our LSTM models, allowing for up-to-the-minute market analysis and prediction, boosting decision efficiency by 30%.</li><li><strong>Risk Management:</strong> Integrating live data with risk metrics like VaR and Sortino Ratio ensures that our system can react swiftly to market volatility, helping to lower simulated investment risk by 22%.</li><li><strong>Competitive Edge:</strong> The ability to process and act on data faster than competitors can lead to significant gains in simulated portfolio returns.</li></ul><p>The engineering challenge lay in ensuring low-latency data flow and high throughput. By leveraging distributed streaming platforms, we created a resilient and efficient backbone for our predictive models, proving that robust data engineering is foundational to successful AI applications in finance.</p>`
    },
    {
        title: "Interpretable AI: Beyond the Black Box",
        date: "2025-06-08",
        image: "https://images.unsplash.com/photo-1581093589118-aa7827806f1d?auto=format&fit=crop&w=1080&q=80",
        tags: ["AI Ethics", "Explainable AI", "Machine Learning"],
        content: `<p>As AI models become increasingly complex, especially in critical applications like predictive maintenance, understanding *why* a model makes a certain prediction is as important as the prediction itself. This is where Explainable AI (XAI) comes into play. In my SMART Predictive Maintenance System, I focused on making the hybrid Deep Learning and Reinforcement Learning model transparent using <strong>SHAP and LIME</strong>.</p><p>SHAP (SHapley Additive exPlanations) provides a unified framework for interpreting predictions, assigning an importance value to each feature for a particular prediction. LIME (Local Interpretable Model-agnostic Explanations) explains the predictions of any classifier or regressor by approximating it with a local, interpretable model.</p><p>By integrating these tools, we were able to:</p><ul><li><strong>Identify Critical Factors:</strong> Pinpoint the top 5 factors (e.g., specific sensor readings, operating conditions) that most influenced a prediction of equipment failure.</li><li><strong>Build Trust:</b> Provide maintenance teams with clear reasons behind predicted failures, increasing their trust in the AI system.</li><li><strong>Improve Models:</strong> Insights from SHAP and LIME helped in refining features and model architecture, contributing to the 30% improvement in prediction accuracy.</li></ul><p>This commitment to interpretability ensures that our AI solutions are not just powerful but also actionable and trustworthy, bridging the gap between complex algorithms and real-world operational decisions.</p>`
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
                        <!-- Imbalanced dots -->
                        <g class="churn-dot-imbalanced">
                            ${[...Array(6)].map((_, i) => `<circle cx="${40 + i * 40}" cy="50" r="5" fill="#008080" class="churn-dot"/>`).join('')}
                            <circle cx="120" cy="100" r="5" fill="#ffbf00" class="churn-dot"/>
                        </g>
                        <!-- Synthetic dots -->
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
    class Particle { constructor(x, y, dx, dy) { this.x = x; this.y = y; this.directionX = dx; this.directionY = dy; this.size = (Math.random() * 2) + 1; } draw() { ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.fillStyle = 'rgba(139, 92, 246, 0.2)'; ctx.fill(); } update() { if (this.x > bgCanvas.width || this.x < 0) this.directionX = -this.directionX; if (this.y > bgCanvas.height || this.y < 0) this.directionY = -this.directionY; let dx = mouse.x - this.x; let dy = mouse.y - this.y; if (Math.hypot(dx, dy) < mouse.radius + this.size) { if (mouse.x < this.x && this.x < bgCanvas.width - this.size * 10) this.x += 5; if (mouse.x > this.x && this.x > this.size * 10) this.x -= 5; if (mouse.y < this.y && this.y < bgCanvas.height - this.size * 10) this.y += 5; if (mouse.y > this.y && this.y > this.size * 10) this.y -= 5; } this.x += this.directionX; this.y += this.directionY; this.draw(); } }
    function initParticles() { particlesArray = []; let num = (bgCanvas.height * bgCanvas.width) / 9000; for (let i = 0; i < num; i++) { let x = Math.random() * innerWidth; let y = Math.random() * innerHeight; let dx = (Math.random() * .4) - 0.2; let dy = (Math.random() * .4) - 0.2; particlesArray.push(new Particle(x, y, dx, dy)); } }
    function animateParticles() { requestAnimationFrame(animateParticles); ctx.clearRect(0, 0, innerWidth, innerHeight); particlesArray.forEach(p => p.update()); connectParticles(); }
    function connectParticles() { let opacityValue = 1; for (let a = 0; a < particlesArray.length; a++) { for (let b = a; b < particlesArray.length; b++) { let distance = Math.hypot(particlesArray[a].x - particlesArray[b].x, particlesArray[a].y - particlesArray[b].y); if (distance < 120) { opacityValue = 1 - (distance / 120); ctx.strokeStyle = `rgba(167, 139, 250, ${opacityValue * 0.3})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(particlesArray[a].x, particlesArray[a].y); ctx.lineTo(particlesArray[b].x, particlesArray[b].y); ctx.stroke(); } } } }
    resizeHandler();
    animateParticles();
});
