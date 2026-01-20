def build_result(metadata, model_info, final, scores, timeline):
    return {
        "metadata": metadata,
        "model": model_info,
        "predictions": {
            "final_instruments": final,
            "confidence_scores": scores,
            "timelines": timeline
        }
    }
