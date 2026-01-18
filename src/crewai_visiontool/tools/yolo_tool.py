from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from ultralytics import YOLO

class YoloToolInput(BaseModel):
    """Input schema for YoloTool."""
    image_path: str = Field(..., description="The absolute path to the image file to analyze.")

class YoloTool(BaseTool):
    name: str = "Object Detection Tool"
    description: str = (
        "A tool that uses the YOLOv8 model to detect objects in an image. "
        "It takes an image path as input and returns a list of detected objects and their counts."
    )
    args_schema: Type[BaseModel] = YoloToolInput
    _model: YOLO = None

    def _run(self, image_path: str) -> str:
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if self._model is None:
                model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
                self._model = YOLO(model_name)  # Load the YOLO model once
            
            results = self._model(image_path, verbose=False)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self._model.names[class_id]
                    detected_objects.append(class_name)
            
            # Count occurrences of each object
            from collections import Counter
            object_counts = Counter(detected_objects)
            
            # Format output
            output_lines = ["Detected objects:"]
            for obj, count in object_counts.items():
                output_lines.append(f"- {obj}: {count}")
                
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
