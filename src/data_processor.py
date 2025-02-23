import pandas as pd

class DataProcessor:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
    
    def clean_data(self) -> pd.DataFrame:
        self.data = self.data.dropna().sort_values('date')
        return self.data
    
    def get_latest_data(self, periods: int = None) -> pd.DataFrame:
        return self.data.tail(periods) if periods else self.data