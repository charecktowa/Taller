from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()


# Define the Item model
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


# In-memory "database"
items = []


# Create an item (POST)
@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    # Check if item with the same id already exists
    for existing_item in items:
        if existing_item.id == item.id:
            raise HTTPException(
                status_code=400, detail="Item with this ID already exists"
            )
    items.append(item)
    return item


# Read all items (GET)
@app.get("/items/", response_model=List[Item])
async def read_items():
    return items


# Read a single item by ID (GET)
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    for item in items:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


# Update an item (PUT)
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, updated_item: Item):
    for index, item in enumerate(items):
        if item.id == item_id:
            items[index] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")


# Delete an item (DELETE)
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    for index, item in enumerate(items):
        if item.id == item_id:
            items.pop(index)
            return {"detail": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")
