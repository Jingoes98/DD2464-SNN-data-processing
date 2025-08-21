import torch
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sequence_length, overlap_length, batch_size = 75, 25, 16


class EventDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        target = self.targets[idx]
        return frame, target

def create_sequences(data, sequence_length, overlap_length, batch_size):
    sequences = []
    start_index = 0
    
    while start_index + sequence_length <= len(data):
        sequence = data[start_index:start_index + sequence_length]
        sequences.append(sequence)
        start_index += (sequence_length - overlap_length)
    
    num_sequences = (len(sequences) // batch_size) * batch_size
    sequences = sequences[:num_sequences]
    
    return sequences

def get_data(number):
    try:
        data = torch.load(f"training_data/{number}.pt")
        if not data[0].is_coalesced():
            data[0] = data[0].coalesce()
        
        if not data[1].is_coalesced():
            data[1] = data[1].coalesce()
            
        frames_tensor = data[0].to_dense().float()
        targets  = data[1].to_dense().float()
                
        frames_sequence = create_sequences(frames_tensor, sequence_length, overlap_length, batch_size) # torch.Size([536, 50, 200, 200])
        target_sequence = create_sequences(targets, sequence_length, overlap_length, batch_size) # torch.Size([536, 50, 64, 64])
        
        X_train, X_test, y_train, y_test = train_test_split(frames_sequence, target_sequence, test_size=0.2, random_state=42)

        dataset = EventDataset(X_train, y_train)
        test_dataset = EventDataset(X_test, y_test)
        
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
        
        #print(len(frames_tensor))
        #print(len(frames_sequence))
        return train_loader, test_loader
    
    except Exception as e:
        print(f"Error loading file {number}.pt: {e}")
        return None
    
