import argparse
import pandas as pd
import time
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ============================================================
# 🔹 Base Class
# ============================================================
class StringMatcher:
    def search_count(self, text: str, pattern: str) -> int:
        """Mengembalikan jumlah kemunculan pola dalam teks."""
        raise NotImplementedError

    def search_indices(self, text: str, pattern: str) -> list[int]:
        """Mengembalikan daftar indeks awal dari setiap kemunculan pola."""
        raise NotImplementedError


# ============================================================
# 🔹 KMP Algorithm
# ============================================================
class KMPMatcher(StringMatcher):
    def _compute_lps(self, pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    def search_indices(self, text, pattern):
        if not pattern:
            return []
        lps = self._compute_lps(pattern)
        i = j = 0
        indices = []
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == len(pattern):
                    indices.append(i - j)
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return indices

    def search_count(self, text, pattern):
        return len(self.search_indices(text, pattern))


# ============================================================
# 🔹 Rabin–Karp Algorithm
# ============================================================
class RabinKarpMatcher(StringMatcher):
    def search_indices(self, text, pattern, prime=101):
        if not pattern:
            return []
        m, n = len(pattern), len(text)
        if m > n:
            return []
        d = 256
        h = pow(d, m - 1) % prime
        p = t = 0
        indices = []

        for i in range(m):
            p = (d * p + ord(pattern[i])) % prime
            t = (d * t + ord(text[i])) % prime

        for s in range(n - m + 1):
            if p == t and text[s:s + m] == pattern:
                indices.append(s)
            if s < n - m:
                t = (d * (t - ord(text[s]) * h) + ord(text[s + m])) % prime
                if t < 0:
                    t += prime
        return indices

    def search_count(self, text, pattern):
        return len(self.search_indices(text, pattern))


# ============================================================
# 🔹 Boyer–Moore Algorithm
# ============================================================
class BoyerMooreMatcher(StringMatcher):
    def _bad_char_table(self, pattern):
        table = [-1] * 256
        for i, c in enumerate(pattern):
            table[ord(c)] = i
        return table

    def search_indices(self, text, pattern):
        if not pattern:
            return []
        m, n = len(pattern), len(text)
        bad_char = self._bad_char_table(pattern)
        indices = []
        s = 0

        while s <= n - m:
            j = m - 1
            while j >= 0 and pattern[j] == text[s + j]:
                j -= 1
            if j < 0:
                indices.append(s)
                s += m - bad_char[ord(text[s + m])] if s + m < n else 1
            else:
                s += max(1, j - bad_char[ord(text[s + j])])
        return indices

    def search_count(self, text, pattern):
        return len(self.search_indices(text, pattern))


# ============================================================
# 🔹 File Processing & Profiling
# ============================================================
def process_file_all_algorithms(
    file_csv,
    kata_kunci,
    output_csv="output_results.csv",
    output_md="output_report.md",
    chunksize=10000,
    targetcol="Text"
):
    algorithms = {
        "KMP": KMPMatcher(),
        "RabinKarp": RabinKarpMatcher(),
        "BoyerMoore": BoyerMooreMatcher()
    }

    kata = kata_kunci.lower()
    rows_out = []
    total_rows = 0
    runtime_stats = {name: 0.0 for name in algorithms}
    total_counts = {name: 0 for name in algorithms}
    occurrence_buckets = {
        name: {0:0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, ">5": 0} for name in algorithms
    }

    print(f"🚀 Memulai profiling untuk string search: '{kata_kunci}' menggunakan semua algoritma pada kolom '{targetcol}'...")

    # 🔹 Baca CSV dalam chunk
    for i, chunk in enumerate(pd.read_csv(file_csv, chunksize=chunksize)):
        if targetcol not in chunk.columns:
            raise KeyError(f"Kolom '{targetcol}' tidak ditemukan dalam file CSV.")
        
        print(f"🔍 Memproses chunk ke-{i+1} (jumlah baris: {len(chunk)})...")
        chunk[targetcol] = chunk[targetcol].fillna('').astype(str)

        for text in chunk[targetcol]:
            total_rows += 1
            text_lower = text.lower()

            result_row = {"original_text": text_lower}

            for name, algo in algorithms.items():
                t0 = time.perf_counter()
                indices = algo.search_indices(text_lower, kata)
                count = len(indices)
                t1 = time.perf_counter()

                runtime_ms = (t1 - t0) * 1000.0
                runtime_stats[name] += runtime_ms
                total_counts[name] += count

                # 🔹 Hitung bucket distribusi jumlah kemunculan
                if count >= 6:
                    occurrence_buckets[name][">5"] += 1
                else:
                    occurrence_buckets[name][count] += 1

                result_row[f"{name}_count"] = count
                result_row[f"{name}_indices"] = indices
                result_row[f"{name}_time_ms"] = round(runtime_ms, 4)

            rows_out.append(result_row)

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(output_csv, index=False, encoding='utf-8')

    # ========================================================
    # 🔹 Buat laporan Markdown
    # ========================================================
    md = []
    md.append("# 📊 Laporan Profiling String Matching\n\n")
    md.append(f"**Kata kunci:** `{kata_kunci}`\n\n")
    md.append(f"**Total baris teks yang diuji:** {total_rows}\n\n")

    md.append("## ⏱️ Rata-rata & Total Runtime (ms)\n")
    md.append("| Algoritma | Total Runtime (ms) | Rata-rata per Teks (ms) | Total Kemunculan |\n")
    md.append("|------------|------------------:|------------------------:|-----------------:|\n")
    
    for name in algorithms:
        total_time = runtime_stats[name]
        avg_time = total_time / total_rows if total_rows else 0
        total_count = total_counts[name]
        md.append(f"| {name} | {total_time:.2f} | {avg_time:.6f} | {total_count} |\n")
    
    # ========================================================
    # 🔹 Distribusi jumlah kemunculan
    # ========================================================
    md.append("\n## 📈 Distribusi Jumlah Kemunculan Pola per Algoritma\n")
    md.append("| Algoritma | 0x | 1x | 2x | 3x | 4x | 5x | >5x |\n")
    md.append("|------------|----:|----:|----:|----:|----:|----:|----:|\n")

    for name in algorithms:
        b = occurrence_buckets[name]
        md.append(f"| {name} | {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]} | {b[5]} | {b['>5']} |\n")

    md.append("\n## 📁 File Output\n")
    md.append(f"- CSV: `{output_csv}`\n")
    md.append(f"- Laporan ini: `{output_md}`\n\n")

    with open(output_md, "w", encoding="utf-8") as f:
        f.writelines(md)

    print(f"\n✅ Profiling selesai. Hasil tersimpan di:\n- {output_csv}\n- {output_md}")

    return {
        "csv": output_csv,
        "md": output_md,
        "stats": {
            "total_rows": total_rows,
            "total_counts": total_counts,
            "runtime_stats": runtime_stats
        }
    }


# ============================================================
# 🔹 Contoh Penggunaan
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profiling tiga algoritma string matching (KMP, Rabin-Karp, Boyer-Moore)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Reviews.csv",
        help="Nama file CSV yang akan dianalisis (default: Reviews.csv)"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="Text",
        help="Nama kolom dalam CSV yang akan dicocokkan string-nya (default: 'Text')"
    )

    parser.add_argument(
        "--search",
        type=str,
        default="good",
        help="Kata kunci yang ingin dicocokkan dalam teks (default: 'good')"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=10000,
        help="Ukuran chunk pembacaan CSV untuk efisiensi (default: 10000)"
    )

    args = parser.parse_args()
    file_dataset = args.dataset
    kata_kunci = args.search
    process_file_all_algorithms(
        file_dataset,
        kata_kunci,
        output_csv=f"Reviews-search-for-{kata_kunci}-profiling_{timestamp}.csv",
        output_md=f"Reviews-search-for-{kata_kunci}-profiling_{timestamp}.md",
        targetcol=args.target,
        chunksize=args.chunk
    )
