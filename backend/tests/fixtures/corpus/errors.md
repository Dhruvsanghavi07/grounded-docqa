# Handling Errors { #handling-errors }

To return HTTP responses with errors to the client you use `HTTPException`.

## Raise an HTTPException { #raise }

Raise an `HTTPException` with a status code of 404 when an item is not found. The client receives a JSON body with a `detail` key.

```python
raise HTTPException(status_code=404, detail="Item not found")
```
