# Ballet Motion Analyzer - Architecture Design

## System Overview

```
+-------------------+       +-------------------+       +-------------------+
|   React Frontend  | <---> |  FastAPI Backend   | <---> |  Analysis Engine  |
|                   |       |                   |       |                   |
| - Video Upload    |  REST | - /analyze        |       | - MediaPipe       |
| - Canvas Overlay  |  API  | - /feedback       |       |   BlazePose       |
| - Result Display  |       | - WebSocket       |       | - Ballet Metrics  |
|                   |       |   (real-time)     |       | - Scoring Engine  |
+-------------------+       +-------------------+       +-------------------+
```

## Module Structure

```
ballet-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── pose_estimator.py       # MediaPipe wrapper
│   │   ├── ballet_metrics.py       # Ballet-specific angle/score calculations
│   │   ├── scoring_engine.py       # Pro-standard comparison & advice
│   │   └── overlay_renderer.py     # Server-side overlay generation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── VideoUploader.tsx
│   │   │   ├── SkeletonOverlay.tsx  # Canvas-based overlay rendering
│   │   │   ├── MetricsPanel.tsx     # Angle/score dashboard
│   │   │   └── AdviceCard.tsx       # Improvement suggestions
│   │   └── hooks/
│   │       └── useAnalysis.ts       # API communication hook
│   └── package.json
└── ARCHITECTURE.md
```

## Data Flow

1. User uploads video/image -> Frontend sends to `/api/analyze`
2. Backend extracts frames (1-5 fps for lightweight processing)
3. MediaPipe BlazePose processes each frame -> 33 landmark coordinates
4. `ballet_metrics.py` computes ballet-specific angles and metrics
5. `scoring_engine.py` compares against professional standards
6. Results streamed back via WebSocket or returned as JSON
7. Frontend renders skeleton overlay + metrics on Canvas

## Key Design Decisions

- **Lightweight Processing**: Frame sampling at 1-5 fps instead of full framerate
- **Client-side Overlay**: Canvas-based rendering for responsive visual feedback
- **Stateless API**: Each analysis request is independent
- **Progressive Enhancement**: Static image analysis first, video streaming later
