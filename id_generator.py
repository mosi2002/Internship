import uuid

def uuid_id_generator():
        new_uuid = uuid.uuid4()
        return f"p{new_uuid.hex[:8]}"