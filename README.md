# 🌊 POSEIDON: Oceanographic Data Analysis System

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/adi9336/POSEIDON?style=social)](https://github.com/adi9336/POSEIDON/stargazers)

POSEIDON is an advanced oceanographic data analysis system that processes and analyzes Argo float data to provide meaningful insights about ocean conditions. The system uses natural language processing to understand user queries and generates SQL queries to retrieve relevant data from a database.

$screenshotSection = @"

## 📸 Screenshots

<div align=""center"">
  <img src=""screenshots/Screenshot 2025-11-23 120743.png"" alt=""POSEIDON Query Interface"" style=""max-width: 90%; border: 1px solid #ddd; border-radius: 4px; padding: 5px;"">
</div>

"@

# Read the current README content
$readmeContent = Get-Content -Path .\README.md -Raw

# Insert the screenshot section after the Features section
$updatedReadme = $readmeContent -replace "(## 🌟 Features[^#]*)##", "`$1`n`n$screenshotSection`n##"

# Save the updated content back to README.md
$updatedReadme | Set-Content -Path .\README.md -Encoding UTF8`

## 🌟 Features

- **Natural Language Processing**: Understands complex oceanographic queries
- **Data Retrieval**: Fetches data from Argo floats based on location, depth, and time
- **SQL Generation**: Automatically generates optimized SQL queries
- **Data Analysis**: Provides statistical insights and summaries
- **Interactive**: Easy-to-use command-line interface

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adi9336/POSEIDON.git
   cd POSEIDON
   ```

2. **Set up the environment**
   ```bash
   # Install Poetry if you haven't already
   pip install poetry
   
   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🛠 Usage

### Running the Application

```bash
# Run the main application
python src/agent/Retrieving_Agent.py

# Or run with a specific query
python src/agent/Retrieving_Agent.py "What is the temperature at 500m depth near Mumbai in January 2024?"
```

### Example Queries

- "Show me temperature data from the last 30 days near Hawaii"
- "What's the average salinity at 1000m depth in the Pacific Ocean?"
- "Find all measurements from Argo float WMO_1902671"
- "Compare temperature trends between 2023 and 2024 in the Atlantic"

## 🏗 Project Structure

```
POSEIDON/
├── data/                    # Data storage directory
├── src/
│   ├── agent/               # Main agent and workflow code
│   ├── state/               # State management and models
│   └── tools/               # Data processing and utility functions
├── tests/                   # Unit and integration tests
├── .env.example            # Example environment variables
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Argo Program for providing the oceanographic data
- OpenAI for the language models
- All contributors who have helped improve this project

## 📧 Contact

For any questions or suggestions, please open an issue or contact the maintainers.

---

<div align="center">
  Made with ❤️ by the POSEIDON team
</div>
