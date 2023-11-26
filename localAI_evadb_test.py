import evadb
from strategy_model import predict_user_interests, num_items
import collections
import redis
import numpy as np


class CustomCache:
    def __init__(self, capacity=50, model_based_capacity=40, model=None, db_manager=None):
        """ Initialize the custom cache with specified capacities and instances. """
        self.capacity = capacity
        self.model_based_capacity = model_based_capacity
        self.model_based_cache = set()
        self.fifo_cache = collections.deque()
        self.redis_conn = redis.Redis(host='localhost', port=6379, db=0)
        self.model = model
        self.request_count = 0
        self.db_manager = db_manager  # Instance of the database manager

    def update_cache_with_model(self):
        """ Update the cache based on the model prediction. """
        if self.model and self.db_manager:
            # Get prediction results using the database manager
            predicted_items = self.db_manager.get_predict_res()

            # Update the model prediction cache area
            self.model_based_cache.clear()  # Clear the current cache
            for item in predicted_items:
                self.model_based_cache.add(item)
                self.redis_conn.set(f"model_item_{item}", f"data_for_{item}")

    def update_fifo_cache(self, item):
        """ Update the FIFO cache. """
        self.fifo_cache.append(item)
        if len(self.fifo_cache) > self.capacity - self.model_based_capacity:
            removed_item = self.fifo_cache.popleft()
            self.redis_conn.delete(f"fifo_item_{removed_item}")

    def process_request(self, id, data):
        """ Process each request by inserting data into the history table and updating the cache. """
        # Insert request data into the History database table
        self.db_manager.insert_to_history(id, data)

        # Update FIFO cache
        self.update_fifo_cache(data)

        # Update the cache with the model every 100 requests
        self.request_count += 1
        if self.request_count % 100 == 0:
            self.update_cache_with_model()


class EvaDBManager:
    def __init__(self):
        """ Initialize database connection and cursor. """
        self.conn = evadb.connect()
        self.cursor = self.conn.cursor()

    def query(self, sql):
        """ Execute a SQL query and return the result as a DataFrame. """
        return self.cursor.query(sql).df()

    def create_function(self):
        """ Create a custom function in the database if it doesn't exist. """
        self.query("""
            CREATE FUNCTION
            IF NOT EXISTS BufferPredict
            IMPL 'buffer_strategy.py';
        """)

    def setup_database(self):
        """ Setup the database by creating necessary tables. """
        self.query("DROP TABLE IF EXISTS History")
        self.query("""
            CREATE TABLE History
            (id INTEGER,
            data TEXT(30));
        """)

    def insert_to_history(self, id, data):
        """ Insert data into the History table. """
        self.query(f"""
            INSERT INTO History (id, data) VALUES
            ('{id}', '{data}');
        """)

    def get_predict_res(self):
        """ Get prediction results based on the recent data in the History table. """
        # Fetch the last 1000 records from the History table
        recent_data_query = """
            SELECT * FROM History
            ORDER BY id DESC
            LIMIT 1000;
        """
        recent_data = self.query(recent_data_query)

        # Assume a BufferPredict function exists in the database for prediction
        predict_query = f"""
            SELECT BufferPredict({recent_data}).result FROM dual;
        """
        predict_result = self.query(predict_query)

        # Handle the prediction results
        predicted_items = predict_result['result'].iloc[0]

        return predicted_items

    def list_all_functions(self):
        """ List all functions available in the database. """
        return self.query("SHOW FUNCTIONS;")

    def cleanup(self):
        """ Cleanup the database by dropping functions. """
        self.query("DROP FUNCTION BufferPredict")


def test_cache_hit_rate(cache, requests):
    """ Test the cache hit rate for a series of requests. """
    hits = 0
    total = len(requests)

    for req in requests:
        cache.process_request(req)

        # Check if the request hits the cache
        if f"model_item_{req}" in cache.model_based_cache or \
                f"fifo_item_{req}" in cache.fifo_cache:
            hits += 1

    hit_rate = hits / total
    return hit_rate


def main():
    """ Main function to test the cache strategy. """
    # Assuming num_items and model are defined
    # Initialize database manager instance
    db_manager = EvaDBManager()
    db_manager.setup_database()  # Setup the database

    # Create an instance of CustomCache
    cache = CustomCache(capacity=50, model_based_capacity=40, model=model, db_manager=db_manager)

    # Simulate 1000 random requests
    requests = np.random.randint(0, num_items, 1000)

    # Calculate the cache hit rate
    hit_rate = test_cache_hit_rate(cache, requests)
    print(f"Cache hit rate: {hit_rate:.2f}")


if __name__ == "__main__":
    main()
