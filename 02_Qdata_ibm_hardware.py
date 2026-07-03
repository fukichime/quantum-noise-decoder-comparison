IBM_TOKEN = "TOKEN_HERE"
BACKEND   = "ibm_marrakesh"   
NUM_SHOTS = 1_000
ROUNDS    = 5
SHOTS_PER_JOB = 300            


import math, time
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session


def build_surface_code_circuit(rounds=5):
    data    = QuantumRegister(9, name='data')
    ancilla = QuantumRegister(8, name='ancilla')
    c_syndrome = [ClassicalRegister(8, name=f'round_{r}') for r in range(rounds)]
    c_final    = ClassicalRegister(9, name='final_data')
    qc = QuantumCircuit(data, ancilla, *c_syndrome, c_final)

    stabilizer_map = [
        (0, [0,1,3,4]),
        (2, [1,2,4,5]),
        (4, [3,4,6,7]),
        (6, [4,5,7,8]),
        (1, [0,3]),
        (3, [1,2,4,5]),
        (5, [3,4,6,7]),
        (7, [5,8])
    ]

    for r in range(rounds):
        qc.reset(ancilla)
        for a_idx, _ in stabilizer_map:
            if a_idx % 2 == 0: qc.h(ancilla[a_idx])
        for a_idx, neighbors in stabilizer_map:
            for q_idx in neighbors:
                if a_idx % 2 == 0: qc.cx(ancilla[a_idx], data[q_idx])
                else:               qc.cx(data[q_idx], ancilla[a_idx])
        for a_idx, _ in stabilizer_map:
            if a_idx % 2 == 0: qc.h(ancilla[a_idx])
        for a_idx in range(8):
            qc.measure(ancilla[a_idx], c_syndrome[r][a_idx])
        qc.barrier()

    for q_idx in range(9):
        qc.measure(data[q_idx], c_final[q_idx])

    return qc


def extract_features_and_labels(data_bin, rounds):
    import numpy as np

    def unpack(arr, num_bits):
        unpacked = np.unpackbits(arr.astype(np.uint8), axis=1)
        return unpacked[:, :num_bits]

    final_arr  = unpack(data_bin.final_data.array, 9)
    round_arrs = [unpack(getattr(data_bin, f'round_{r}').array, 8)
                  for r in range(rounds)]

    num_shots = final_arr.shape[0]
    features  = []
    labels    = []

    for i in range(num_shots):
        syndrome_bits = []
        for r in range(rounds):
            row = round_arrs[r][i]
            syndrome_bits.extend([int(b) for b in reversed(row)])
        features.append(syndrome_bits)

        fd = final_arr[i]
        q0 = int(fd[-1])
        q1 = int(fd[-2])
        q2 = int(fd[-3])
        labels.append(q0 ^ q1 ^ q2)

    return features, labels

def run_on_ibm(token, backend_name, total_shots, shots_per_job, rounds):
    print("Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel  = "ibm_cloud",
        token    = token,
        instance = "INSTANCE HERE"
    )
    backend = service.backend(backend_name)
    print(f"  Backend : {backend.name}  ({backend.num_qubits} qubits)")

    print("Building and transpiling circuit...")
    qc   = build_surface_code_circuit(rounds=rounds)
    qc_t = transpile(qc, backend=backend, optimization_level=1)
    print(f"  Transpiled depth : {qc_t.depth()},  gates : {qc_t.size()}")

    num_jobs     = math.ceil(total_shots / shots_per_job)
    all_features = []
    all_labels   = []

    print(f"\nRunning {total_shots} shots across {num_jobs} jobs "
          f"({shots_per_job} shots each)...")
    print("Track jobs at: quantum.ibm.com → Jobs\n")

    sampler = Sampler(mode=backend)

    for job_idx in range(num_jobs):
        this_shots = min(shots_per_job, total_shots - job_idx * shots_per_job)
        job = sampler.run([qc_t], shots=this_shots)
        print(f"  Job {job_idx+1:>3}/{num_jobs}  id={job.job_id()}  "
              f"shots={this_shots}  status={job.status()}")

        while job.status() not in ("DONE", "ERROR", "CANCELLED"):
            time.sleep(15)
            print(f"             ... {job.status()}")

        if job.status() != "DONE":
            raise RuntimeError(f"Job {job.job_id()} failed: {job.status()}")

        data_bin = job.result()[0].data
        feat, lab = extract_features_and_labels(data_bin, rounds)
        all_features.extend(feat)
        all_labels.extend(lab)
        print(f"             → {len(feat)} shots parsed  "
              f"(total: {len(all_features)})")

    return all_features, all_labels

def sanity_check_against_aer(rounds=5, shots=500):

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        print("qiskit-aer not installed — skipping sanity check")
        return

    print("\n── Sanity check (Aer, 500 shots) ──")
    qc = build_surface_code_circuit(rounds=rounds)

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h'])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])

    sim   = AerSimulator(method='stabilizer')
    from qiskit import transpile as tp
    result = sim.run(tp(qc, sim), noise_model=noise, shots=shots, memory=True).result()
    memory = result.get_memory()

    aer_features, aer_labels = [], []
    for shot_string in memory:
        parts = shot_string.split()
        syndrome_bits = []
        for round_str in reversed(parts[1:]):
            syndrome_bits.extend([int(b) for b in reversed(round_str)])
        aer_features.append(syndrome_bits)
        fd  = parts[0]
        aer_labels.append(int(fd[-1]) ^ int(fd[-2]) ^ int(fd[-3]))

    import numpy as np

    class FakeBitArray:
        def __init__(self, arr): self.array = arr

    class FakeDataBin:
        pass

    db  = FakeDataBin()
    n   = len(memory)
    final_mat   = np.zeros((n, 9),  dtype=np.uint8)
    round_mats  = [np.zeros((n, 8), dtype=np.uint8) for _ in range(rounds)]

    for i, shot_string in enumerate(memory):
        parts = shot_string.split()
        for bit_idx, ch in enumerate(parts[0]):
            final_mat[i, bit_idx] = int(ch)
        for r in range(rounds):
            round_str = parts[rounds - r]   
            for bit_idx, ch in enumerate(round_str):
                round_mats[r][i, bit_idx] = int(ch)

    db.final_data = FakeBitArray(final_mat)
    for r in range(rounds):
        setattr(db, f'round_{r}', FakeBitArray(round_mats[r]))

    ibm_features, ibm_labels = extract_features_and_labels(db, rounds)

    # Compare
    match_features = all(a == b for a, b in zip(aer_features, ibm_features))
    match_labels   = all(a == b for a, b in zip(aer_labels,   ibm_labels))

    print(f"  Feature vectors match : {match_features}")
    print(f"  Labels match          : {match_labels}")
    if match_features and match_labels:
        print("  ✓ Bit ordering verified — safe to run on IBM hardware")
    else:
        print("  ✗ MISMATCH — do not proceed until this is resolved")
        for idx, (af, ibf) in enumerate(zip(aer_features, ibm_features)):
            if af != ibf:
                print(f"    First mismatch at shot {idx}:")
                print(f"    Aer : {af}")
                print(f"    IBM : {ibf}")
                break

    return match_features and match_labels


if __name__ == "__main__":
    if IBM_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        raise ValueError(
            "\nSet IBM_TOKEN at the top of this file.\n"
            "Get it from: quantum.ibm.com → profile → API token\n"
        )

    ok = True

    print(f"\nGenerating distance-3 circuit (IBM real device)...")
    features, labels = run_on_ibm(
        token         = IBM_TOKEN,
        backend_name  = BACKEND,
        total_shots   = NUM_SHOTS,
        shots_per_job = SHOTS_PER_JOB,
        rounds        = ROUNDS,
    )

    pd.DataFrame(features).to_csv('features_realdevice.csv', index=False)
    pd.DataFrame(labels, columns=['label']).to_csv('labels_realdevice.csv', index=False)

    print("\nData generation complete!")
    print("Features shape (real device):", pd.read_csv('features_realdevice.csv').shape)
    print("Labels shape   (real device):", pd.read_csv('labels_realdevice.csv').shape)
    print("\nReal-device label distribution:")
    dist = pd.read_csv('labels_realdevice.csv')['label'].value_counts()
    print(dist)
    print(f"\nLogical error rate: {labels.count(1)/len(labels):.4f} "
          f"({labels.count(1)/len(labels)*100:.1f}%)")
    print("\nDone.")
