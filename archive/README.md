# **DeepSeek vs. ChatGPT: AI Performance & User Behavior Dataset**  

## **Overview**  
This dataset is a **synthetically generated AI performance benchmark** that simulates real-world interactions across industries, regions, and devices. It provides insights into **user engagement, AI accuracy, response speed, and adoption trends** for both **ChatGPT (GPT-4-turbo) and DeepSeek (DeepSeek-Chat 1.5).**  


### **Key Features:**  
- **AI performance metrics** (accuracy, response time)  
- **User engagement trends** (churn, retention, active users)  
- **AI usage across industries** (education, debugging, content creation, etc.)  
- **Geographic distribution** of AI adoption  
- **Device preferences** (mobile, desktop, smart devices)  


> **Designed for AI Benchmarking, Business Intelligence, Machine Learning, and Data Visualization (Power BI, SQL, Python).**  


---

## **Dataset Details**  

- **File**: `deepseek_vs_chatgpt.csv`  
- **Rows**: 10,000+ AI interactions  
- **Columns**: 27 features  

### **Sample Columns**  
| Column Name         | Description |
|---------------------|-------------|
| `Date`             | Date of AI interaction |
| `AI_Platform`      | AI model used (ChatGPT or DeepSeek) |
| `AI_Model_Version` | Model version (e.g., GPT-4-turbo, DeepSeek-Chat 1.5) |
| `Active_Users`     | Simulated active users per platform per day |
| `New_Users`        | Number of new users joining the platform |
| `Churned_Users`    | Users who stopped using the platform |
| `Retention_Rate`   | Percentage of returning users |
| `Query_Type`       | Type of AI request (Technical, General, etc.) |
| `Response_Accuracy` | AI response correctness score (0-1) |
| `Response_Speed_sec` | AI response time in seconds |
| `Device_Type`      | Mobile, Desktop, Tablet, or Smart Speaker |
| `Region`          | Simulated user country/region |

📌 **For the full list of columns, visit the dataset on Kaggle.**  

---

## **Disclaimer: This is Synthetic Data**  
This dataset was **artificially generated** using **Python (`faker` library) and SQL** to create a **realistic but fictional AI usage dataset**.  

- **No real user data was collected**—this dataset is strictly for **educational, analytical, and visualization purposes**.  
- It represents **general AI trends** but is **not sourced from actual AI user logs**.  

---

## **Can You Access Real DeepSeek or ChatGPT Data?**  

### **DeepSeek Data Access**  
To access **real DeepSeek data**, you need to use the **DeepSeek API**, which requires signing up for an API key. This allows you to connect your code to their models using the **OpenAI SDK**.  

⚠️ **Privacy Considerations:**  
- DeepSeek **stores all user data on servers in China**.  
- There are concerns about potential data access by **Chinese government authorities**.  
- **Before using the DeepSeek API, ensure you understand the privacy risks.**  

🔗 **More info:** [DeepSeek API Documentation](https://deepseek.com/)  

---

### **ChatGPT Data Access**  
No, you **cannot directly access** global user data for ChatGPT.  

- **Why?** OpenAI prioritizes user privacy and does not share **comprehensive individual user data**.  
- However, **aggregated insights** can be analyzed using:  
  - **Website traffic statistics**  
  - **Regional usage patterns**  
  - **Search trends** (SimilarWeb, Google Trends, etc.)  

While these methods provide **some level of insight**, **detailed ChatGPT user behavior data remains private**.  

🔗 **More info:** [OpenAI Privacy Policy](https://openai.com/privacy/)  

---

## **Use Cases**  
🔍 **AI Performance Benchmarking** – Compare DeepSeek vs. ChatGPT’s response accuracy & speed.  
📈 **Business Intelligence** – Analyze AI adoption across industries and regions.  
📊 **Data Visualization** – Build Power BI dashboards for AI trends.  
🤖 **Machine Learning Models** – Predict AI response quality based on query type.  
🛠 **Product Optimization** – Improve AI user experience and engagement.  

> **Explore AI trends and drive data-driven decisions!**  

---

## **How to Use This Dataset**  

### **1️⃣ Load Data in Python (Pandas)**
```python
import pandas as pd

df = pd.read_csv("deepseek_vs_chatgpt.csv")
print(df.head())
```

### **2️⃣ Run SQL Queries in PostgreSQL**
```sql
SELECT AI_Platform, 
       AVG(Response_Accuracy) AS Avg_Accuracy, 
       AVG(Response_Speed_sec) AS Avg_Speed
FROM deepseek_vs_chatgpt
GROUP BY AI_Platform;
```

### **3️⃣ Create Power BI Dashboards**
- Import `deepseek_vs_chatgpt.csv` into **Power BI**  
- Use **KPI Cards, Area Charts, and Bar Charts** for AI performance trends  

---

## **License & Citation**  
This dataset is open-source under the **MIT License**. Feel free to use it for research, projects, and analysis.  

If you use this dataset, please credit:  
```bibtex
@misc{deepseek_vs_chatgpt_2025,
  title = {DeepSeek vs. ChatGPT: AI Performance & User Behavior Dataset},
  author = {Your Name or Kaggle Username},
  year = {2025},
  url = {https://www.kaggle.com/datasets/your-dataset-link}
}
```

---

## **Feedback & Contributions**  
-  **Found this dataset useful?** Support it with an upvote on Kaggle!  
- **Suggestions?** Feel free to comment, fork, or contribute.  

> **Uploading to Kaggle?**  
> 1️⃣ Save `deepseek_vs_chatgpt.csv` and `README.md` in your dataset folder.  
> 2️⃣ Go to **Kaggle → Datasets → Create New Dataset**.  
> 3️⃣ Upload both files, fill in details, and click **Publish**.  
> 4️⃣ Share the link and showcase your work!  
