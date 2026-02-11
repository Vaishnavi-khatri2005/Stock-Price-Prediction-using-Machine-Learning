"""
START HERE - Your First Steps Guide
Stock Price Prediction ML Project
"""

# ============================================================================
# 👋 START HERE - FIRST TIME USER GUIDE
# ============================================================================

Welcome to the Stock Price Prediction Machine Learning Project!

This document will guide you through your first steps. Read this first!


## 📖 WHAT IS THIS PROJECT?

This is a complete machine learning system that predicts stock price movements 
using historical market data and technical indicators.

It includes:
- Data downloading and preprocessing
- 50+ technical features
- 5 different ML models
- Interactive web dashboard
- Complete documentation
- Working examples


## ⏱️ TIME COMMITMENT

- Reading this guide: 5 minutes
- Installation: 5-10 minutes
- First run: 10-15 minutes
- Total: 20-30 minutes to see results!


## 🎯 YOUR FIRST 5 STEPS

### Step 1: Install Python (2 min)
If you don't have Python 3.8+:
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or higher
3. Run installer, check "Add Python to PATH"
4. Verify: Open Command Prompt, type: python --version

### Step 2: Install Dependencies (5 min)
Open Command Prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

Wait for all packages to install. This includes:
- Data processing (pandas, numpy)
- Machine learning (scikit-learn, xgboost, lightgbm)
- Deep learning (tensorflow)
- Visualization (plotly, matplotlib)
- Dashboard (streamlit)

### Step 3: Run Your First Pipeline (15 min)
Execute the quick start script:

```bash
python quickstart.py
```

Follow the prompts (press Enter to use defaults):
- Stock ticker: AAPL
- Years of data: 5
- Prediction days: 5

Watch it:
- Download data
- Engineer features
- Train 5 models
- Generate results

### Step 4: View Your Results (2 min)
After completion, check:
- `visualizations/` folder - See generated plots
- Console output - Model performance metrics
- Check specific files:
  - predictions_comparison.png
  - feature_importance.png
  - model_comparison.png

### Step 5: Try the Dashboard (5 min)
For interactive exploration:

```bash
streamlit run dashboard.py
```

This opens a web interface where you can:
- Load different stocks
- Engineer features visually
- Train models interactively
- See real-time results


## 🔧 THREE WAYS TO USE THIS PROJECT

### Option A: Quick Command Line (Simplest)
```bash
python quickstart.py
```
- Follow prompts
- Get results in 15 minutes
- No coding needed
- Best for quick testing

**Choose this if**: You want results fast and don't need to modify anything.

### Option B: Interactive Dashboard (Most User-Friendly)
```bash
streamlit run dashboard.py
```
- Web-based interface
- Visual step-by-step
- Can modify settings in UI
- No command line needed
- Great for learning

**Choose this if**: You like visual interfaces and want to explore interactively.

### Option C: Python Script (Most Flexible)
```python
from main_pipeline import StockPricePredictionPipeline

pipeline = StockPricePredictionPipeline('MSFT')
pipeline.run_pipeline()
```
- Full control
- Can customize everything
- Requires Python knowledge
- Best for integration/automation

**Choose this if**: You're comfortable with Python and want customization.


## 🎓 LEARNING RESOURCES IN ORDER

### For Quick Understanding (15 min read)
1. **PROJECT_SUMMARY.md** - High-level overview
2. **QUICKREF.md** - Command examples

### For Complete Understanding (1-2 hours)
1. **README.md** - Comprehensive guide
2. **notebooks/complete_example.py** - Step-by-step walkthrough
3. **Code comments** - In each Python file

### For Deep Dive (2-4 hours)
1. **SETUP.md** - Installation details
2. **INDEX.md** - File-by-file guide
3. **Source code** - Read module code
4. **config.py** - Understand customization

### For Mastery (4+ hours)
1. Study each module deeply
2. Modify and experiment
3. Add new features
4. Deploy your own version


## ❓ COMMON QUESTIONS

**Q: Do I need to know Python?**
A: No for quickstart.py or dashboard.py. Yes if you want to modify code.

**Q: How much data do I need?**
A: Project downloads automatically from Yahoo Finance (5 years by default).

**Q: Which stocks can I use?**
A: Any stock ticker (AAPL, MSFT, GOOGL, etc.).

**Q: Can I use my own data?**
A: Yes! Place CSV in data/ folder and load it.

**Q: What if installation fails?**
A: Read SETUP.md troubleshooting section or check error messages.

**Q: How long does it take to run?**
A: 10-20 minutes on normal computer (varies by stock/timeframe).

**Q: Can I make money with this?**
A: This is for learning. Don't use for real trading without more testing.

**Q: What are the results?**
A: See visualizations/ folder for generated plots and metrics.


## 🚨 TROUBLESHOOTING - QUICK FIXES

### Problem: "ModuleNotFoundError"
```bash
# Solution:
pip install -r requirements.txt
```

### Problem: "Connection timeout"
```bash
# Solution: Check internet, then try again
# Or use existing CSV file
```

### Problem: "Python not found"
```bash
# Solution: Install Python from python.org
# Make sure "Add to PATH" is checked
```

### Problem: Slow execution
```bash
# Solution: This is normal on first run
# Subsequent runs with same data are faster
# Consider reducing data size in config.py
```

### Problem: Out of memory
```bash
# Solution: Use smaller dataset
# Change in config.py or command line
```

For more solutions, see SETUP.md.


## 📁 IMPORTANT FILES TO KNOW

### Must Know (for first-time use)
- **quickstart.py** - Run this first!
- **dashboard.py** - Visual interface
- **requirements.txt** - Install dependencies

### Should Know (for understanding)
- **main_pipeline.py** - Main orchestrator
- **data_loader.py** - Gets stock data
- **feature_engineering.py** - Creates features
- **model_training.py** - ML models
- **README.md** - Full documentation

### Nice to Know (for advanced use)
- **config.py** - Customization
- **utils.py** - Helper tools
- **notebooks/complete_example.py** - Examples


## 🎯 YOUR LEARNING JOURNEY

```
Day 1: Quick Start
├─ Install Python
├─ pip install -r requirements.txt
├─ python quickstart.py
└─ See results in visualizations/

Day 2: Understand
├─ Read README.md
├─ Try dashboard.py
├─ Review notebooks/complete_example.py
└─ Understand each component

Day 3: Customize
├─ Modify config.py
├─ Try different stocks
├─ Adjust parameters
└─ Run pipeline multiple times

Day 4: Extend
├─ Add new features
├─ Implement new models
├─ Try ensemble approaches
└─ Integrate with other code

Week 2: Master
├─ Deep understand ML concepts
├─ Optimize performance
├─ Deploy as application
└─ Apply to real problems
```


## 💡 TIPS FOR SUCCESS

### Before Running
1. Ensure internet connection (for data download)
2. Have 2GB free disk space
3. Check Python version (3.8+)

### First Run
1. Use defaults (AAPL, 5 years)
2. Let it complete fully
3. Check visualizations/ folder
4. Read output messages

### Troubleshooting
1. Read error messages carefully
2. Check SETUP.md section
3. Try smaller dataset
4. Verify installation

### Next Steps
1. Try different stocks
2. Read complete documentation
3. Experiment with parameters
4. Customize for your needs


## 🎓 WHAT YOU'LL LEARN

By completing this project, you'll understand:
- ✓ Machine learning pipelines
- ✓ Feature engineering
- ✓ Multiple ML algorithms
- ✓ Time-series analysis
- ✓ Data preprocessing
- ✓ Model evaluation
- ✓ Web dashboards
- ✓ Python best practices


## 📊 WHAT TO EXPECT

### After First Run:
- 5 trained models
- Performance metrics
- Generated visualizations
- Feature importance analysis
- Prediction charts

### Typical Results:
- RMSE: 0.01-0.05 (depends on model)
- R²: 0.3-0.6 (reasonable for stock prediction)
- Feature importance: 5-10 key indicators
- Accuracy: ~50-65% directional


## 🚀 READY TO START?

Here's your quick checklist:

- [ ] Python 3.8+ installed
- [ ] In project directory
- [ ] Ran: pip install -r requirements.txt
- [ ] Ran: python quickstart.py
- [ ] Checked visualizations/ folder
- [ ] Read README.md

**All done? You're ready!** 🎉


## 📞 GETTING HELP

If stuck:
1. Re-read this document
2. Check SETUP.md
3. Read README.md
4. Review error messages
5. Check code comments
6. Try example code


## 🎉 FINAL WORDS

You now have everything needed to:
✓ Run a complete ML pipeline
✓ Train 5 different models
✓ Generate visualizations
✓ Understand the process
✓ Customize the system
✓ Extend with new features

**The best way to learn is by doing!**

Start with quickstart.py, explore results, then dive deeper.


## 📚 DOCUMENTATION MAP

If you need:
- Quick overview → PROJECT_SUMMARY.md
- Commands & code → QUICKREF.md
- Installation help → SETUP.md
- Complete guide → README.md
- File navigation → INDEX.md
- Step-by-step learning → notebooks/complete_example.py
- Customization → config.py
- Help troubleshooting → SETUP.md


## 🌟 NEXT STEP

**Now go run this command:**

```bash
python quickstart.py
```

Then explore your results!

---

**Happy Machine Learning! 🚀📊**

Questions? Check the documentation files.
Ready for more? Review the README.md.

Good luck! 🎯
