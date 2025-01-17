import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from titans_transformer import TitansTransformer
import time
import torch.nn as nn


def get_data(subset_size=1000):
    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    # Tokenize the dataset and limit the size
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Select a subset of the dataset
    train_data = tokenized_dataset['train'].select(range(subset_size))
    val_data = tokenized_dataset['validation'].select(range(subset_size // 10))  # Smaller validation set
    test_data = tokenized_dataset['test'].select(range(subset_size // 10))       # Smaller test set

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data['input_ids'], dtype=torch.long)
    val_data = torch.tensor(val_data['input_ids'], dtype=torch.long)
    test_data = torch.tensor(test_data['input_ids'], dtype=torch.long)

    vocab_size = tokenizer.vocab_size

    return train_data, val_data, test_data, vocab_size


def batchify(data, batch_size, device):
    # Divide data into batch_size parts
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def evaluate(model, data_source, criterion, batch_size, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, batch_size):
            data = data_source[i:i + batch_size].to(device)
            targets = data_source[i + 1:i + 1 + batch_size].to(device)
            # Ensure input and target sizes match
            if data.size(0) != targets.size(0):
                break  # Skip incomplete batch
            output = model(data)
            total_loss += criterion(output.view(-1, output.size(-1)), targets.view(-1)).item()
    return total_loss / (data_source.size(0) - 1)


def train_model():
    # Hyperparameters
    batch_size = 16
    eval_batch_size = 10
    d_model = 512
    nhead = 8
    num_layers = 6
    memory_size = 512
    epochs = 3  # Fewer epochs for testing
    subset_size = 1000  # Limit dataset size for testing

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data (subset for testing)
    train_data, val_data, test_data, vocab_size = get_data(subset_size=subset_size)

    # Initialize model
    model = TitansTransformer(
        num_tokens=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        memory_size=memory_size
    ).to(device)

    # Batchify data
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Train
        model.train()
        total_loss = 0.
        for batch, i in enumerate(range(0, train_data.size(0) - 1, batch_size)):
            data = train_data[i:i + batch_size].to(device)
            targets = train_data[i + 1:i + 1 + batch_size].to(device)

            # Ensure input and target sizes match
            if data.size(0) != targets.size(0):
                break  # Skip incomplete batch

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            if batch % 100 == 0:
                curr_loss = total_loss / (batch + 1)
                print(f'| epoch {epoch + 1:3d} | batch {batch:3d} | '
                      f'loss {curr_loss:5.2f}')


        # Evaluate
        val_loss = evaluate(model, val_data, criterion, eval_batch_size, device)
        print('-' * 89)
        print(f'| end of epoch {epoch + 1:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
              f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'titans_transformer_model.pt')

        scheduler.step()

    # Test
    model.load_state_dict(torch.load('titans_transformer_model.pt'))
    test_loss = evaluate(model, test_data, criterion, eval_batch_size, device)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f}')
    print('=' * 89)


if __name__ == "__main__":
    train_model()
