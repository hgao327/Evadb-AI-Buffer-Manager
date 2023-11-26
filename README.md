# Evadb-AI-Buffer-Manager
EvaDB offers an AI-integrated database solution, while LocalAI enables high-speed execution of AI models locally. Combining these technologies, we propose the development of an AI-Buffer-Manager designed to optimize caching strategies by intelligently analyzing user behavior and data access patterns, thereby improving the overall performance of database systems.

### Structure:

<img width="695" alt="Screenshot 2023-11-25 at 8 16 53 PM" src="https://github.com/hgao327/Evadb-AI-Buffer-Manager/assets/108708761/af529ad2-c509-4983-a0eb-548998e7a3b0">

#### 

### *Attention

The predictive capability of deep learning models is related to data access patterns; different browsing habits, application scenarios, or usage methods will result in different replacement strategies. For example, in a movie recommendation system, the data in Redis can be determined based on multiple recall strategies. If at the database system level, different access habits will lead to different outcomes, hence causing fluctuations in the results. In the future, it may be necessary to further study the use of more generalized models with a broader range of applicability or to optimize specifically according to different scenarios.

## - Implementation details

### 1. CustomCache Class

#### Initialization

```python
class CustomCache:
    def __init__(self, capacity=50, model_based_capacity=40, model=None, db_manager=None):
        ...
```

- **Capacity Settings**: The cache is initialized with `capacity`, which is the total number of items it can hold, and `model_based_capacity`, which is the number of items in the cache managed based on the model's predictions.
- **Model and DB Manager**: The `model` is a predictive model, potentially a machine learning model, for anticipating future cache demands. `db_manager` is an instance of `EvaDBManager` for database interactions.

#### Model-Based Cache Updating

```python
def update_cache_with_model(self):
    if self.model and self.db_manager:
        predicted_items = self.db_manager.get_predict_res()
        ...
```

- This method clears the current model-based cache and updates it with items predicted by the model.

#### FIFO Cache Updating

```python
def update_fifo_cache(self, item):
    self.fifo_cache.append(item)
    if len(self.fifo_cache) > self.capacity - self.model_based_capacity:
        ...
```

- Manages a FIFO cache by appending new items. When the cache exceeds its capacity, the oldest item is removed.

#### Request Processing

```python
def process_request(self, id, data):
    self.db_manager.insert_to_history(id, data)
    self.update_fifo_cache(data)
    ...
```

- Processes incoming requests, stores them in a database for historical data, and updates the FIFO cache accordingly. Model-based cache is updated every 100th request.



### 2. Class EvaDBManager

```python
pythonCopy code
class EvaDBManager:
    """ A class to manage database operations, particularly for caching strategies. """
```

This class is designed to handle various database operations, crucial for implementing and managing caching strategies.

#### Initialization Method

```python
pythonCopy code
def __init__(self):
    """ Initialize database connection and cursor. """
    self.conn = evadb.connect()
    self.cursor = self.conn.cursor()
```

- Initializes the database connection using `evadb.connect()` and sets up a cursor for executing SQL commands.

#### Query Execution Method

```python
pythonCopy code
def query(self, sql):
    """ Execute a SQL query and return the result as a DataFrame. """
    return self.cursor.query(sql).df()
```

- Executes a given SQL command and returns the result as a DataFrame, a format that is convenient for data manipulation and analysis.

#### Function Creation Method

```python
pythonCopy code
def create_function(self):
    """ Create a custom function in the database if it doesn't exist. """
    self.query("""
        CREATE FUNCTION
        IF NOT EXISTS BufferPredict
        IMPL 'buffer_strategy.py';
    """)
```

- Creates a custom function named `BufferPredict` in the database if it doesn't already exist. This function is likely linked to a caching strategy, possibly involving predictive algorithms.

#### Database Setup Method

```python
pythonCopy code
def setup_database(self):
    """ Setup the database by creating necessary tables. """
    self.query("DROP TABLE IF EXISTS History")
    self.query("""
        CREATE TABLE History
        (id INTEGER,
        data TEXT(30));
    """)
```

- Sets up the database by creating necessary tables, particularly a `History` table, which can be used to store data relevant to caching operations or AI model training.

#### Data Insertion Method

```python
pythonCopy code
def insert_to_history(self, id, data):
    """ Insert data into the History table. """
    self.query(f"""
        INSERT INTO History (id, data) VALUES
        ('{id}', '{data}');
    """)
```

- Inserts data into the `History` table. This method is crucial for keeping a record of data changes and user requests.

#### Prediction Results Retrieval Method

```python
pythonCopy code
def get_predict_res(self):
    """ Get prediction results based on the recent data in the History table. """
    recent_data_query = """
        SELECT * FROM History
        ORDER BY id DESC
        LIMIT 1000;
    """
    recent_data = self.query(recent_data_query)

    predict_query = f"""
        SELECT BufferPredict({recent_data}).result FROM dual;
    """
    predict_result = self.query(predict_query)
    predicted_items = predict_result['result'].iloc[0]
    return predicted_items
```

- Retrieves prediction results using the latest data from the `History` table. It assumes the existence of the `BufferPredict` function in the database for making predictions, possibly about which items should be cached next.

#### Listing Functions Method

```python
pythonCopy code
def list_all_functions(self):
    """ List all functions available in the database. """
    return self.query("SHOW FUNCTIONS;")
```

- Lists all available functions in the database, useful for managing and debugging custom functions in the database.

#### Cleanup Method

```python
pythonCopy code
def cleanup(self):
    """ Cleanup the database by dropping functions. """
    self.query("DROP FUNCTION BufferPredict")
```

- Cleans up the database, particularly by removing functions that are no longer in use.

Overall, the `EvaDBManager` class is a comprehensive tool for database management, particularly suited for caching strategies that involve interactions with a database, such as predictive caching and data preprocessing. By integrating AI models with database operations, this class efficiently manages cache data while providing necessary support to optimize caching strategies.



### 3. Cache Hit Rate Testing Function

```python
def test_cache_hit_rate(cache, requests):
    hits, total = 0, len(requests)
    for req in requests:
        cache.process_request(req)
        ...
```

- Simulates a series of requests, processing each one through the cache, and calculates the cache hit rate.



### 4. Main Function

```python
def main():
    db_manager = EvaDBManager()
    db_manager.setup_database()
    cache = CustomCache(capacity=50, model_based_capacity=40, model=model, db_manager=db_manager)
    requests = np.random.randint(0, num_items, 1000)
    hit_rate = test_cache_hit_rate(cache, requests)
    print(f"Cache hit rate: {hit_rate:.2f}")
```

- Initializes and sets up the database, creates a cache instance, simulates requests, and evaluates the cache hit rate.

#### Detailed Workflow

1. **Cache Initialization**: The `CustomCache` class initializes the cache with specified capacities. It maintains two cache sections: a model-based cache and a FIFO cache.

2. **Handling Requests**: Each request is processed by storing its data in the `History` table and updating the FIFO cache. The model-based cache is updated after every 100 requests, using predictions from the model.

3. **Database Management**: `EvaDBManager` handles database connections, data insertion, and retrieving data for predictions.

4. **Evaluating Cache Performance**: The `test_cache_hit_rate` function assesses the effectiveness of the caching strategy by simulating requests and calculating the hit rate.

This approach exemplifies an advanced cache management strategy, leveraging predictive modeling to enhance cache performance alongside traditional caching techniques. The integration of a machine learning model for predictive caching demonstrates a proactive approach to data retrieval and cache management.



### 5. User Interest Prediction Model Using LSTM and Attention Mechanism

This script defines a machine learning model using TensorFlow and Keras, aimed at predicting user interests based on their browsing history.

#### Data Preparation

- **Simulating User Data**: Generates simulated user browsing histories. Each user has a history of 10 items, and the model aims to predict the next item of interest.
- **Preprocessing**: User histories are padded for uniformity, a standard procedure in sequential data processing for machine learning.

#### Model Building (`build_model` function)

- Model Architecture

  : The model's architecture is comprised of:

  - An `Embedding` layer for input data transformation.
  - A `Bidirectional LSTM` layer for learning patterns in sequential data.
  - An `Attention` mechanism for focusing on specific parts of the input sequence.
  - `Dense` layers, including a dropout layer to prevent overfitting.
  - A softmax activation function in the output layer for classification.

- **Compilation**: The model is compiled with a loss function and optimizer suited for classification tasks.

#### Making Predictions (`predict_user_interests` function)

- **Prediction Process**: Processes a user's browsing history and predicts their interests using the trained model.
- **Top Predictions Extraction**: Identifies the top `n` items that the user is likely to be interested in next.

#### Example Usage

- The model is created, trained, and then used to predict the top interests for a given user's browsing history.

This implementation showcases the use of LSTM and attention mechanisms in neural networks for sequential data analysis and future behavior prediction, particularly useful in recommendation systems.



### **6. EvaDB Registration**

```python
class BufferPredict(AbstractFunction):
    @setup(cacheable=False, function_type="predict_replacement", batchable=False)
    def setup(self):
        """Initial setup for the BufferPredict function. Cacheable, function type, and batchable are defined."""
        pass

    @property
    def name(self) -> str:
        """Define the name of the function."""
        return "Predict"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["command"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["result"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward the request to an external API and process the response."""

        # Extract input data from the first row of the dataframe
        input_data = df.iloc[0, 0]

        # Define the API URL and headers
        url = "http://localhost:8080/v1/buffer/predict"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": input_data}

        # Make a POST request to the external API
        response = requests.post(url, headers=headers, json=data)
        response_data = json.loads(response.text)

        # Assume the response data is a list, where each element contains a 'data' field
        # Extract data from the first 40 elements
        data_extracted = [item["data"] for item in response_data["data"][:40]]

        # Place the extracted results into a DataFrame
        df = pd.DataFrame({'result': data_extracted})
        return df
```



### **7. Cache Replacement Strategy for comparison**

#### LRUCache

```python
pythonCopy code
class LRUCache:
    """ Least Recently Used (LRU) cache strategy """
```

- This class implements the LRU caching strategy where the least recently accessed item is replaced when the cache reaches its capacity. It uses an `OrderedDict` to maintain the order of item usage.

#### LFUCache

```python
pythonCopy code
class LFUCache:
    """ Least Frequently Used (LFU) cache strategy """
```

- LFU caching strategy prioritizes eviction based on item access frequency. This class uses a dictionary to store the items and a `Counter` to keep track of how often each item is accessed.

#### RandomCache

```python
pythonCopy code
class RandomCache:
    """ Random replacement cache strategy """
```

- In the random replacement strategy, items are evicted at random when new items need to be added to a full cache. This class uses a `set` to store the cached items.

#### FIFOCache

```python
pythonCopy code
class FIFOCache:
    """ First-In-First-Out cache strategy """
```

- FIFO strategy evicts the oldest cached item first. This class also uses a `set` for storage and a `deque` to track the order in which items were added.

#### MRUCache

```python
pythonCopy code
class MRUCache:
    """ Most Recently Used cache strategy """
```

- The MRU strategy is the opposite of LRU; the most recently used item is evicted first. It uses a simple dictionary to maintain the cache.

#### SHiPCache

```python
pythonCopy code
class SHiPCache:
    """ SHiP (Signature History based Predictor) cache strategy """
```

- SHiP is a more complex strategy that uses historical access signatures to predict which items will be needed soon. It maintains a main cache and a history set to manage predictions and actual cached items.

### Testing Function

```python
pythonCopy code
def test_cache_strategy(cache_class, capacity, requests):
    ...
```

- This function tests the hit rate of a given cache replacement strategy. It creates an instance of the cache class, processes a series of requests, and calculates the proportion of hits to total requests to determine effectiveness.

Each of these classes and the testing function can be used to simulate and measure the performance of different cache replacement strategies. By observing the hit rates from these simulations, one can analyze the suitability of each strategy for specific use cases, such as web applications, database systems, or content delivery networks. The hit rate is a critical performance metric that influences how quickly users receive data and can significantly affect the user experience and system efficiency.



## - Sample output / Metrics measuring

<img width="480" alt="Screenshot 2023-11-25 at 8 16 26 PM" src="https://github.com/hgao327/Evadb-AI-Buffer-Manager/assets/108708761/e9548f8f-0ecf-4df9-98ca-a3a320558cde">

**EvaDB AI-Manager**: With the highest hit rate of 0.3166, the AI-Manager outperforms all other strategies. This demonstrates the effectiveness of using AI to predict which items will be needed soon. The AI-Manager's predictive capabilities allow it to preemptively cache items that are more likely to be accessed, significantly increasing the chances of cache hits.

The use of an AI-based strategy in this case has significantly outperformed traditional cache replacement strategies. It suggests that integrating machine learning to understand and predict user behavior or access patterns can lead to more efficient cache utilization. However, the effectiveness of such a strategy would also depend on the quality of the AI model and the nature of the data and access patterns. Traditional methods still play a role, particularly when the access pattern is less predictable or when the AI model cannot be invoked frequently due to computational overheads or latency constraints.
