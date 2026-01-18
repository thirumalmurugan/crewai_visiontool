# CrewAI Vision Tool with YOLO and Local Llama

Welcome to the **CrewaiVisiontool** project. This project coordinates an AI agent that analyzes images using a **YOLOv8** model and reports findings using a local **Llama** model powered by **Ollama**.

## Features
- **Object Detection**: Uses YOLOv8 (nano) for fast and accurate object identification.
- **Local LLM Integration**: Uses Ollama to run `llama3.2` for interpreting detections and generating reports.
- **Privacy-First**: Operates entirely on your local machine once models are downloaded.
- **Optimized Performance**: Features model caching and quiet mode to reduce console clutter.
- **Safety Measures**: Includes a `max_iter` limit to prevent agent loops with smaller local models.

## Prerequisites
- **Python**: >=3.10 <3.14
- **Ollama**: [Download and install Ollama](https://ollama.com/)
- **Local Model**: Pull the default model: `ollama pull llama3.2`

## Installation

We recommend using the CrewAI CLI for installation to manage the virtual environment correctly:

```bash
# Install the project and dependencies
crewai install
```

### Dependency Note
This project requires specific versions of `litellm` and `openai` to avoid resolution conflicts. If you are installing manually with `pip`, please refer to `pyproject.toml` for the pinned versions.

## Configuration

Update your `.env` file in the project root:

```env
MODEL=ollama/llama3.2
OLLAMA_BASE_URL=http://localhost:11434
YOLO_MODEL=yolov8n.pt
LITELLM_LOG=OFF
```

### Configuration Details:
- `MODEL`: The local model name (prefix with `ollama/`).
- `OLLAMA_BASE_URL`: Usually `http://localhost:11434`.
- `YOLO_MODEL`: The YOLO model file (e.g., `yolov8n.pt`).
- `LITELLM_LOG`: Set to `OFF` to suppress internal LiteLLM telemetry and logging.

## Running the Project

### Using CrewAI CLI (Default)
To run with the default sample image:
```bash
crewai run
```

### Using Python Module (Specific Image)
To analyze a specific image file:
```bash
python -m crewai_visiontool.main "path/to/your/image.jpg"
```

## Troubleshooting

### "Manual Interrupt" or Loops
If the agent repeats tool calls, we have added `max_iter=3` in `crew.py` to stop it. We have also refined the agent's backstory in `agents.yaml` to encourage immediate reporting.

### "ModuleNotFoundError"
If you see missing modules (like `litellm` or `fastapi_sso`), ensured you have run `crewai install` or `uv sync`. The project uses a local `.venv` by default.

## Directory Structure
- `src/crewai_visiontool/tools/yolo_tool.py`: Custom YOLOv8 tool with caching.
- `src/crewai_visiontool/config/`: YAML files for Agent and Task definitions.
- `src/crewai_visiontool/crew.py`: Core logic and LLM configuration.
- `src/crewai_visiontool/main.py`: CLI entry point.

## Support
- [crewAI Documentation](https://docs.crewai.com)
- [Ultralytics (YOLO) Documentation](https://docs.ultralytics.com/)
- [Ollama Documentation](https://ollama.com/library)
