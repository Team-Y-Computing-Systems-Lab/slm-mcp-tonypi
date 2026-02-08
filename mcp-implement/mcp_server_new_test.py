def summarize_scene():
    """Use DINO detections to describe the scene."""
    try:
        url = VISION_API_URL
        params = {
            "request": "everything",
            "boundaryColors": ""
        }

        resp = requests.post(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        detections = data.get("detections", [])

        if not detections:
            return {
                "status": "success",
                "summary": "I don't clearly see any object in view."
            }

        # Compute approximate image width
        max_xmax = max(det.get("box", {}).get("xmax", 0) for det in detections)
        img_w = data.get("image_width", max_xmax if max_xmax > 0 else 640)

        # Convert detections into position phrases
        phrases = []

        
        # Build natural-language summary
        if len(phrases) == 1:
            summary = f"I see {phrases[0]}."
        else:
            summary = "I see " + ", ".join(phrases[:-1]) + " and " + phrases[-1] + "."

        return {"status": "success", "summary": summary}

    except Exception as e:
        print(f"Summarize Scene failed: {e}")
        return {"status": "error", "error": str(e)}
    
    
    
def summarize_scene():
    """Use chat API to describe the scene based on DINO detections."""
    try:
        url = VISION_API_URL
        params = {
            "request": "red box;green box;blue box",
            "boundaryColors": "255,0,0;0,255,0;0,0,255"
        }
        print(f"\n[DEBUG only] Calling DINO API: {url}")
        print(f"[DEBUG only] Params: {params}")
        
        resp = requests.post(url, params=params, timeout=30)
        print(f"[DEBUG only] Status: {resp.status_code}")
        print(f"[DEBUG only] Headers: {dict(resp.headers)}")
        print(f"[DEBUG only] Raw response (only the first 500 characters): {resp.text[:500]}")
        if resp.status_code != 200:
            return {"status": "success", "summary": "Vision system unavailable."}
        
        try:
            data = resp.json()
            print(f"[DEBUG only] Parsed JSON: {json.dumps(data, indent=2)[:1000]}")

            detections = []
            positions = []
            
            if "detections" in data and isinstance(data["detections"], list):
                for det in data["detections"]:
                    if isinstance(det, dict):
                        label = det.get("label", "object")
                        detections.append(label)
                        
                        # get the object position from the bounding box
                        box = det.get("box", {})
                        if isinstance(box, dict):
                            xmin = box.get("xmin", 0)
                            xmax = box.get("xmax", 640)
                            center_x = (xmin + xmax) / 2
                            
                            # determine the position with respect to the image frame
                            if center_x < 213:
                                positions.append("left")
                            elif center_x > 427:
                                positions.append("right")
                            else:
                                positions.append("center")
            
            if not detections:
                return {"status": "success", "summary": "I don't see any clear objects."}
            
            # format for LLM with positions
            if len(detections) == len(positions):
                # combine object + position data
                object_descriptions = []
                for i in range(len(detections)):
                    object_descriptions.append(f"{detections[i]} on the {positions[i]}")
                
                detection_text = ", ".join(object_descriptions)
            else:
                detection_text = ", ".join(detections)
            
            chat_url = VISION_API_URL.rsplit("/", 1)[0] + "/chat_api"
            prompt = f"""
            The robot sees: {', '.join(detection_text)}            
            Please provide a short, natural description starting with "I see...". Keep it to one sentence. There should be NO ACTION GROUP called"""
            
            chat_resp = requests.post(chat_url, params={"request": prompt}, timeout=30)
            chat_data = chat_resp.json()
            summary = chat_data.get("response", f"I see {len(detections)} object(s).")
            return {"status": "success", "summary": summary}
        
        except:
            return {
                "status": "success",
                "summary": "Vision system is active. Try 'Capture Image' for specific objects."
            }
        
    except Exception as e:
        print(f"Summarize Scene failed: {e}")
        return {"status": "error", "error": str(e)}