import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from web3 import Web3
from collections import Counter
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import seaborn as sns


# Connect to the Ethereum network
w3 = Web3(Web3.HTTPProvider('https://gensyn-testnet.g.alchemy.com/v2/V7LmcVlfsvOBPfNDQksGn9ceteQDUniw'))

# Get block by number
# block_number = 123456  # Replace with the desired block number or use 'latest'
# block = w3.eth.get_block(block_number)
# print(block)

st.title("Gensyn Testnet Activity Dashboard")

# Slider widget to let user choose how many recent blocks to analyze
N_BLOCKS = st.slider("Number of recent blocks to analyze:", min_value=100, max_value=1000, value=500)

@st.cache_data(show_spinner=False)
def fetch_blocks(n):
    latest_block = w3.eth.block_number
    data = []
    for i in range(latest_block - n + 1, latest_block + 1):
        block = w3.eth.get_block(i, full_transactions=True)
        data.append(block)
        time.sleep(0.05)
    return data

block_data = fetch_blocks(N_BLOCKS)

# 1. Active Addresses
address_counter = Counter()
contract_counter = Counter()

for block in block_data:
    for tx in block.transactions:
        address_counter[tx['from']] += 1
        if tx['to']:
            address_counter[tx['to']] += 1
        if tx['to'] and tx['input'] != '0x':
            contract_counter[tx['to']] += 1

st.header("👥 Top Active Addresses")
top_addresses = address_counter.most_common(10)
st.dataframe(pd.DataFrame(top_addresses, columns=["Address", "Interactions"]))

# 2. Active Users
user_counter = Counter()

for block in block_data:
    for tx in block.transactions:
        user_counter[tx['from']] += 1
        if tx['to']:
            user_counter[tx['to']] += 1

top_users = []
for address, count in user_counter.most_common():
    # Check if the address is not a contract
    if w3.eth.get_code(address) == b'':
        top_users.append((address, count))
    if len(top_users) == 10:
        break

st.header("🧑‍💻 Top Users")
st.dataframe(pd.DataFrame(top_users, columns=["User Address", "Interactions"]))

# 3. Active Contracts & Unique Users per Contract
contract_counter = Counter()
contract_user_map = defaultdict(set)

for block in block_data:
    for tx in block.transactions:
        if tx['to'] and tx['input'] != '0x':
            contract_counter[tx['to']] += 1
            if w3.eth.get_code(tx['to']) != b'':  # Check if it's a contract
                contract_user_map[tx['to']].add(tx['from'])

calls_df = pd.DataFrame(contract_counter.items(), columns=["Contract Address", "Calls"])
users_df = pd.DataFrame([(c, len(u)) for c, u in contract_user_map.items()], columns=["Contract Address", "Unique Users"])

combined_df = pd.merge(calls_df, users_df, on="Contract Address")
combined_df["Calls per User"] = combined_df["Calls"] / combined_df["Unique Users"]
combined_df = combined_df.sort_values(by="Calls", ascending=False)

st.header("📊 Top Smart Contracts")
st.dataframe(combined_df.head(10))

# 4. Identify Most Suspicious Contract
suspicious_users = combined_df[(combined_df["Unique Users"] <= 2) & (combined_df["Calls per User"] > 100)]

if not suspicious_users.empty:
    suspicious_contract = suspicious_users.iloc[0]["Contract Address"]

    user_tx_count = defaultdict(int)
    timestamps = []

    for block in block_data:
        for tx in block.transactions:
            if tx['to'] == suspicious_contract:
                user_tx_count[tx['from']] += 1
                timestamps.append(datetime.fromtimestamp(block.timestamp))

    user_tx_list = sorted(user_tx_count.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(user_tx_list, columns=["User", "Txs to Contract"])
    df["Suspicious Contract"] = suspicious_contract  
    df = df[["Suspicious Contract", "User", "Txs to Contract"]]

    st.header("🕵️ Users Interacting with Suspicious Contract")
    st.dataframe(df)

    # Plot time gaps between calls
    timestamps.sort()
    deltas = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]

    st.header("⏱️ Time Gaps Between Calls to Suspicious Contract")
    fig, ax = plt.subplots()
    ax.plot(deltas, marker='o')
    ax.set_xlabel("Call Index")
    ax.set_ylabel("Seconds Since Previous Call")
    ax.set_title("Time Gaps Between Contract Calls")
    st.pyplot(fig)

else:
    st.info("✅ No suspicious contracts detected with current thresholds.")


# 5. Transactions per Block
tx_counts = []
block_times = []

for block in block_data:
    tx_counts.append(len(block.transactions))
    block_times.append(datetime.fromtimestamp(block.timestamp))

# Smooth tx count with 5-block rolling average
tx_counts_smooth = pd.Series(tx_counts).rolling(window=5).mean()

# Z-score
mean_tx = np.mean(tx_counts)
std_tx = np.std(tx_counts)
z_scores = [(count - mean_tx) / std_tx for count in tx_counts]

# Flag suspicious blocks (z > 2)
suspicious_indices = [i for i, z in enumerate(z_scores) if z > 2]
suspicious_times = [block_times[i] for i in suspicious_indices]
suspicious_counts = [tx_counts[i] for i in suspicious_indices]

st.header("📈 Transaction Trends Over Time")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(block_times, tx_counts, alpha=0.4)
ax.plot(block_times, tx_counts_smooth, color='orange')
ax.set_title("Transactions per Block Over Time")
ax.set_xlabel("Block Time")
ax.set_ylabel("Number of Transactions")
ax.scatter(suspicious_times, suspicious_counts, color='red', label="Spike Detected", zorder=5)
ax.legend()
fig.autofmt_xdate(rotation=45)
st.pyplot(fig)

# 6. Gas Efficiency per Block
gas_efficiencies = []
block_times = []

for block in block_data:
    efficiency = block['gasUsed'] / block['gasLimit'] if block['gasLimit'] != 0 else 0
    gas_efficiencies.append(efficiency)
    block_times.append(datetime.fromtimestamp(block['timestamp']))

# Smooth gas efficiency with 5-block rolling average 
smoothed_efficiencies = pd.Series(gas_efficiencies).rolling(window=5).mean()

st.header("⛽ Gas Usage Efficiency per Block")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(block_times, gas_efficiencies, alpha=0.3, label='Raw')
ax.plot(block_times, smoothed_efficiencies, color='blue')
ax.set_title("Gas Usage Efficiency Over Time")
ax.set_xlabel("Block Time")
ax.set_ylabel("Efficiency (gasUsed / gasLimit)")
ax.set_ylim(0, 1.05)
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# Highlight spike window
spike_mask = (np.array(block_times) > datetime(2025, 7, 17, 19, 45)) & (np.array(block_times) < datetime(2025, 7, 17, 19, 50))
spike_efficiencies = np.array(gas_efficiencies)[spike_mask]

spike_indices = np.where(spike_mask)[0]
max_idx = np.argmax(spike_efficiencies)
block_with_max_eff = spike_indices[max_idx]

st.write("📌 Block with Max Gas Efficiency:")
st.write("Block Time:", block_times[block_with_max_eff])
st.write("Gas Efficiency:", gas_efficiencies[block_with_max_eff])

# 7. 
st.header("📊 Correlation: Gas Efficiency vs Transaction Count")

fig, ax = plt.subplots()
ax.scatter(tx_counts, gas_efficiencies, alpha=0.6)
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Gas Usage Efficiency")
ax.set_title("Efficiency vs Transactions per Block")
st.pyplot(fig)

st.caption("Built with 💡 using Alchemy, Web3.py, and Streamlit")