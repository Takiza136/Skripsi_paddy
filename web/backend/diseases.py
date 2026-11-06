"""Informasi singkat tiap kelas penyakit/hama padi (Paddy Doctor, 10 kelas).

Catatan: informasi bersifat umum sebagai edukasi awal. Konfirmasi tetap
disarankan ke penyuluh pertanian / pakar OPT setempat.
"""

DISEASES = {
    "normal": {
        "name": "Daun Sehat",
        "type": "Sehat",
        "severity": "none",
        "cause": "-",
        "description": "Tidak terdeteksi gejala penyakit atau serangan hama pada citra.",
        "symptoms": [
            "Daun berwarna hijau merata",
            "Tidak ada bercak, garis, atau perubahan bentuk daun",
        ],
        "treatment": [
            "Pertahankan pemupukan berimbang (N, P, K)",
            "Lakukan pemantauan rutin setiap 3-7 hari",
            "Jaga kebersihan lahan dari gulma",
        ],
    },
    "bacterial_leaf_blight": {
        "name": "Hawar Daun Bakteri (Kresek)",
        "type": "Penyakit bakteri",
        "severity": "high",
        "cause": "Xanthomonas oryzae pv. oryzae",
        "description": "Bakteri masuk lewat luka atau hidatoda dan menyumbat pembuluh daun sehingga daun mengering dari tepi.",
        "symptoms": [
            "Garis basah kekuningan di tepi daun yang meluas ke arah pangkal",
            "Daun menguning lalu mengering seperti terbakar",
            "Pada semai dapat muncul gejala kresek (daun layu menggulung)",
        ],
        "treatment": [
            "Gunakan varietas tahan dan benih sehat",
            "Hindari pemupukan nitrogen berlebihan",
            "Atur drainase dan hindari genangan terus-menerus",
            "Bersihkan sisa tanaman sakit; bakterisida berbahan aktif tembaga bila perlu",
        ],
    },
    "bacterial_leaf_streak": {
        "name": "Bercak Daun Bergaris Bakteri",
        "type": "Penyakit bakteri",
        "severity": "medium",
        "cause": "Xanthomonas oryzae pv. oryzicola",
        "description": "Bakteri menyerang jaringan di antara tulang daun dan membentuk garis sempit tembus cahaya.",
        "symptoms": [
            "Garis sempit berwarna cokelat kekuningan, sejajar tulang daun",
            "Garis tampak tembus cahaya bila daun diterawang",
            "Garis dapat menyatu hingga daun mengering",
        ],
        "treatment": [
            "Gunakan benih sehat dan varietas yang lebih tahan",
            "Kurangi dosis nitrogen, perbanyak kalium",
            "Jaga sanitasi lahan dan atur jarak tanam",
        ],
    },
    "bacterial_panicle_blight": {
        "name": "Hawar Malai Bakteri",
        "type": "Penyakit bakteri",
        "severity": "high",
        "cause": "Burkholderia glumae",
        "description": "Menyerang malai pada fase pembungaan sehingga bulir gagal terisi.",
        "symptoms": [
            "Bulir pucat atau keabu-abuan dan hampa",
            "Pangkal bulir berwarna cokelat",
            "Malai tegak karena tidak berisi",
        ],
        "treatment": [
            "Gunakan benih bebas patogen dan lakukan perlakuan benih",
            "Hindari nitrogen berlebih dan tanam terlalu rapat",
            "Atur waktu tanam agar pembungaan tidak bertepatan dengan suhu malam tinggi",
        ],
    },
    "blast": {
        "name": "Blas",
        "type": "Penyakit jamur",
        "severity": "high",
        "cause": "Pyricularia oryzae (Magnaporthe oryzae)",
        "description": "Salah satu penyakit padi paling merusak. Dapat menyerang daun, leher malai, dan buku.",
        "symptoms": [
            "Bercak belah ketupat berwarna abu-abu dengan tepi cokelat",
            "Bercak dapat meluas dan menyatu hingga daun mati",
            "Pada leher malai: busuk cokelat dan malai patah",
        ],
        "treatment": [
            "Tanam varietas tahan blas",
            "Hindari pupuk nitrogen berlebihan",
            "Gunakan fungisida yang dianjurkan (mis. berbahan aktif trisiklazol) sesuai petunjuk",
            "Atur jarak tanam dan jaga kelembapan tidak berlebihan",
        ],
    },
    "brown_spot": {
        "name": "Bercak Cokelat",
        "type": "Penyakit jamur",
        "severity": "medium",
        "cause": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "description": "Sering muncul pada tanaman yang kekurangan hara atau tumbuh di lahan kurang subur.",
        "symptoms": [
            "Bercak oval berwarna cokelat dengan titik pusat lebih terang",
            "Bercak tersebar di permukaan daun",
            "Serangan berat membuat daun mengering dan bulir tidak bernas",
        ],
        "treatment": [
            "Perbaiki kesuburan tanah, cukupkan kalium dan silika",
            "Gunakan benih sehat dan perlakuan benih",
            "Aplikasi fungisida bila serangan meluas",
        ],
    },
    "dead_heart": {
        "name": "Sundep (Dead Heart)",
        "type": "Hama (penggerek batang)",
        "severity": "high",
        "cause": "Larva penggerek batang padi (Scirpophaga spp.)",
        "description": "Larva menggerek batang pada fase vegetatif sehingga pucuk daun muda mati.",
        "symptoms": [
            "Daun pucuk menguning lalu mengering dan mudah dicabut",
            "Anakan muda mati meski bagian bawah masih hijau",
            "Terdapat lubang gerekan di pangkal batang",
        ],
        "treatment": [
            "Tanam serempak dan atur pola tanam",
            "Musnahkan tunggul dan sisa tanaman terserang",
            "Pasang perangkap ngengat; insektisida bila melewati ambang pengendalian",
        ],
    },
    "downy_mildew": {
        "name": "Embun Bulu (Downy Mildew)",
        "type": "Penyakit jamur/oomycete",
        "severity": "medium",
        "cause": "Sclerophthora macrospora",
        "description": "Cenderung muncul pada lahan tergenang atau drainase buruk.",
        "symptoms": [
            "Bercak klorotik (kuning pucat) pada daun",
            "Pertumbuhan tanaman kerdil atau abnormal",
            "Daun dapat tampak berpilin atau mengeriting",
        ],
        "treatment": [
            "Perbaiki drainase dan hindari genangan lama",
            "Gunakan benih sehat dan perlakuan benih",
            "Cabut dan musnahkan tanaman yang terserang berat",
        ],
    },
    "hispa": {
        "name": "Hispa (Kumbang Berduri)",
        "type": "Hama",
        "severity": "medium",
        "cause": "Dicladispa armigera",
        "description": "Kumbang dewasa mengikis daun, larva menggerek di dalam jaringan daun.",
        "symptoms": [
            "Garis atau guratan putih memanjang pada daun",
            "Daun tampak transparan dan mengering di bagian yang tergerek",
            "Terlihat kumbang kecil berduri berwarna gelap",
        ],
        "treatment": [
            "Potong ujung daun yang terserang dan musnahkan",
            "Bersihkan gulma di pematang",
            "Insektisida sesuai anjuran bila populasi tinggi",
        ],
    },
    "tungro": {
        "name": "Tungro",
        "type": "Penyakit virus",
        "severity": "high",
        "cause": "Rice tungro virus, ditularkan wereng hijau",
        "description": "Penyakit virus yang menular lewat wereng hijau dan dapat menurunkan hasil sangat drastis.",
        "symptoms": [
            "Daun menguning hingga jingga, dimulai dari ujung daun",
            "Tanaman kerdil dengan jumlah anakan berkurang",
            "Malai kecil dan bulir banyak hampa",
        ],
        "treatment": [
            "Gunakan varietas tahan tungro dan tanam serempak",
            "Kendalikan wereng hijau (pengamatan dan insektisida selektif)",
            "Cabut dan benamkan tanaman sakit; atur pergiliran tanaman",
        ],
    },
}


def get_disease_info(label: str) -> dict:
    """Ambil info penyakit; fallback generik bila label tidak dikenal."""
    return DISEASES.get(label, {
        "name": label.replace("_", " ").title(),
        "type": "-",
        "severity": "medium",
        "cause": "-",
        "description": "Informasi untuk kelas ini belum tersedia.",
        "symptoms": [],
        "treatment": [],
    })
