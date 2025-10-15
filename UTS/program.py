import csv

# Implementasi algoritma Rabin–Karp
def rabin_karp(text, pattern, prime=101):
    """
    Fungsi pencarian substring menggunakan algoritma Rabin–Karp.
    Mengembalikan True jika pattern ditemukan di text, False jika tidak.
    """
    m = len(pattern)
    n = len(text)
    d = 256  # jumlah karakter dalam alfabet (ASCII)

    if m > n:
        return False

    # Hitung hash awal untuk pattern dan substring pertama dari text
    h = pow(d, m - 1) % prime
    p = 0  # hash untuk pattern
    t = 0  # hash untuk text

    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    # Geser window sepanjang text
    for s in range(n - m + 1):
        if p == t:  # jika hash cocok, lakukan pengecekan karakter demi karakter
            if text[s:s + m] == pattern:
                return True

        if s < n - m:
            # Hitung hash berikutnya (rolling hash)
            t = (d * (t - ord(text[s]) * h) + ord(text[s + m])) % prime
            if t < 0:
                t += prime

    return False


# === Bagian utama ===
def deteksi_kata_kunci(file_csv, kata_kunci):
    hasil = []
    with open(file_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            review_text = row['Text']
            ditemukan = rabin_karp(review_text.lower(), kata_kunci.lower())
            hasil.append({
                'Id': row['Id'],
                'ProductId': row['ProductId'],
                'Summary': row['Summary'],
                'Ditemukan': ditemukan
            })
    return hasil


# === Contoh penggunaan ===
if __name__ == "__main__":
    file_dataset = "Reviews.csv"   # ganti dengan path datasetmu
    kata_kunci = "good"

    hasil_deteksi = deteksi_kata_kunci(file_dataset, kata_kunci)

    # Tampilkan 5 hasil pertama
    for data in hasil_deteksi[:5]:
        print(f"ID: {data['Id']} | Summary: {data['Summary']} | Kata kunci ditemukan: {data['Ditemukan']}")
