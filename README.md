# NeuralStyleTransfer

## Setting up the enviornment
1. Create the virualenv pointing to `python3`:

    ```
    virtualenv ./venv -p `which python3` 
    ```
2. Activate this venv with:
    ```
    source venv/bin/activate
    ```

3. Install the requirements:
    ```
   pip install -r requirements.txt
   ```
4. Set up pre-commit hooks for auto-formatting:
    ```bash
    pre-commit install
    ```

