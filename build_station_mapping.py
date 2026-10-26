"""
Builds station_room_mapping.csv: assigns every audio-guide station name (all languages)
to a room / floor of the Van Gogh Museum.

Sources used to decide the room of each stop:
  - Floor plans RV0-RV3.jpg (station labels drawn in each room)
  - 'Current' sheet of 'Audiotour - Changes (EN).xlsx' (stop numbers: 1st digit = floor)
  - Months in which each station appears in the audio-guide data (exhibition stops only appear
    while the exhibition runs: Yellow 13 Feb-17 May 2026, Vincent's Path to Fame from 12 Jun 2026)
Edit the ROOMS / STATIONS dictionaries below if you find a mistake, then re-run.
"""
import csv, os, unicodedata

ROOMS = {  # zone_id: (floor, room_label)
    "R01": ("0", "Room 1 - Self-portraits (Face to face)"),
    "R02": ("1", "Room 2 - Rural life"),
    "R03": ("1", "Room 3 - Farmers / Peasant life"),
    "R04": ("1", "Room 4 - Paris (New perspectives)"),
    "R05": ("1", "Room 5 - Arles (Artistic flourishing)"),
    "R06": ("2", "Room 6 - Temporary presentation"),
    "R07": ("2", "Room 7 - Artistic exchange / Mental health"),
    "R08": ("2", "Room 8 - Family (Brothers and best friends)"),
    "R09": ("3", "Room 9 - Saint-Remy (Painting against all odds)"),
    "R10": ("3", "Room 10 - Auvers (Impassioned nature)"),
    "R11": ("3", "Room 11 - Inspiration (Expression of colour)"),
    "R12": ("3", "Room 12 - The end (Final thoughts)"),
    "EXH": ("EXH", "Exhibition wing - temporary exhibition"),
}

# zone_id -> station names in every language seen in the data. Confidence 'medium' list further below.
STATIONS = {
 "R01": ["Face to face with Van Gogh", "Van Gogh ile yüz yüze", "وجهاً لوجه مع فان كوخ",
         "Self Portrait with Straw Hat", "Hasır Şapkalı Otoportre", "لوحة شخصية مع قبعة من القش",
         "Self Portrait with Pipe and Straw Hat"],
 "R02": ["Rural Life", "Kırsal Hayat", "الحياة الريفية", "Haymaking", "Fienagione", "《制备干草》", "Haciendo heno",
         "La fenaison", "Heuernte", "Fazendo Feno", "헤이메이킹", "Het hooien", "Сенокос", "Saman Yapımı",
         "干し草づくり", "Сінокіс", "صناعة القش", "Léon-Augustin Lhermitte, Haymaking"],
 "R03": ["Painter of Peasant Life", "Kırsal Hayatın Ressamı", "رسام الحياة الريفية", "The Potato Eaters",
         "Patates Yiyenler", "آكلو البطاطا", "The Cottage", "Kulübe", "الكوخ", "Peasant Heads", "Köylü Başı",
         "رؤوس فلاحين", "Congregation Leaving Ref. Church in Nuenen", "Nuenen Reform Kilisesi'nden Çıkan Cemaat",
         "جمع من المصلين يغادرون كنيسة نيونن الإصلاحية", "View of the Sea at Scheveningen",
         "Scheveningen’de Deniz Manzarası", "منظر للبحر في سخيفينيغن", "Still Life with Bible"],
 "R04": ["New Perspectives", "Yeni Bakış Acıları", "رؤى جديدة", "Self Portrait with Grey Felt Hat",
         "Gri Fötr Şapkalı Otoportre", "لوحة شخصية مع قبعة لباد رمادية", "Garden with Courting Couples",
         "Bahçede Kur Yapan Çiftler", "حديقة مع عشاق متواعدين", "Tuin der geliefden: Square Saint-Pierre",
         "Сад із закоханими: площа Сен-П'єр", "Courtesan", "Kurtizan", "غانية",
         "In the Café: Agostina Segatori in Le Tambourin", "Kafede: Le Tambourin’de Agostina Segatori",
         "في المقهى: أغوستينا سيغاتوري في التومبوران", "Boulevard de Clichy", "Clichy Bulvarı", "شارع كليشي",
         "Trees and Undergrowth", "Ağaçlar ve Çalılıklar", "أشجار وشجيرات"],
 "R05": ["Artistic Flourishing", "Sanatsal Gelişim", "إزدهار فني", "Sunflowers", "Zonnebloemen", "Ayçiçekleri",
         "عباد الشمس", "The Bedroom", "Yatak Odası", "غرفة النوم", "Bridge at Langlois", "Langlois Köprüsü",
         "جسر لونغلوا", "The Harvest", "Hasat", "الحصاد", "The Pink Orchard", "Pembe Meyve Bahçesi",
         "البستان الوردي", "Fishing Boats on the Beach at Les Saint-Maries-de-la-Mer",
         "Fishing Boats on the Beach at Les Saintes-Maries-de-la-Mer",
         "Saintes-Maries-de-la-Mer Sahilindeki Balıkçı Tekneleri", "قوارب صيد السمك على شاطئ سانت ماري دو لا مير",
         "Seascape near Les Saint-Marie-de-la-Mer", "Les Saintes-Maries-de-la-Mer Yakınlarında Deniz Manzarası",
         "منظر بحري بالقرب من سانت ماري دو لا مير", "Морський пейзаж у Сен-Марі де-ля-Мер",
         "Gauguin’s Chair", "Gauguin’in Sandalyesi", "كرسي غوغان", "The Zouave", "Zuhaf", "الجندي الجزائري",
         "Portrait of Camille Roulin", "Camille Roulin Portresi", "لوحة شخصية لكامي رولان"],
 "R06": ["Temporary Exhibition", "Sergiler", "المعرض المؤقت"],
 "R07": ["Artistic Exchange", "Sanatsal Etkileşim", "تبادل فني", "The Yellow House", "Het Gele Huis",
         "A Casa Amarela", "Sarı Ev", "البيت الأصفر", "Van Gogh Painting Sunflowers", "Van Gogh che dipinge i girasoli",
         "画向日葵的梵高", "Vincent van Gogh pintando girasoles", "Vincent Van Gogh peignant des tournesols",
         "Sonnenblumen malend", "Vincent van Gogh a pintar girassóis", "해바라기를 그리는 반 고흐",
         "Vincent van Gogh zonnebloemen schilderend", "Ван Гог рисует подсолнухи", "ヒマワリを描くゴッホの肖像",
         "Ayçiçeklerinin Ressamı", "Ван Гог малює соняшники", "فان كوخ يرسم عباد الشمس"],
 "R08": ["Brothers and Best Friends", "Kardeş ve Dost", "الإخوة والأصدقاء المقربون", "Letter", "Mektup", "رسالة",
         "Quinces, Lemons, Pears, and Grapes", "Kweeperen, citroenen, peren en druiven",
         "Ayvalar, Limonlar, Armutlar ve Üzümler", "سفرجل وليمون وكمثرى وعنب",
         # letters (all months -> permanent collection, Family room)
         "Letter from Vincent to Theo", "Lettera da Vincent a Theo", "Carta de Vincent a Theo", "Lettre de Vincent à Théo",
         "书信梵高给提奥的信", "Brief von Vincent an Theo", "Carta de Vincent para Theo", "빈센트가 테오에게 쓴 편지",
         "Brief van Vincent aan Theo", "Письмо от Винсента к Тео", "手紙フィンセントからテオへ", "Vincent’tan Theo’ya Mektup",
         "Лист від Вінсента до Тео", "رسالة من فنسنت إلى تيو",
         "Letter from Theo to Vincent", "Lettera da Theo a Vincent", "Carta de Theo a Vincent", "Lettre de Théo à Vincent",
         "书信 提奥给梵高的信", "Brief von Theo an Vincent", "Carta de Theo para Vincent", "테오가 빈센트에게 쓴 편지",
         "Brief van Theo aan Vincent", "Письмо от Тео к Винсенту", "手紙テオからフィンセントへ", "Theo’dan Vincent’a Mektup",
         "Лист від Тео Вінсентові", "رسالة من تيو إلى فنسنت",
         "Letter from Jo to Vincent", "Lettera da Jo a Vincent", "Carta de Jo a Vincent", "Lettre de Jo à Vincent",
         "书信 乔给梵高的信", "Brief von Jo an Vincent", "Carta de Jo para Vincent", "조가 빈센트에게 쓴 편지",
         "Brief van Jo aan Vincent", "Письмо от Джо к Винсенту", "手紙ヨーからフィンセントへ", "Jo’dan Vincent’a Mektup",
         "Лист від Йоханни Вінсентові", "رسالة من يو إلى فنسنت"],
 "R09": ["Painting Against All Odds", "Herşeye Rağmen Resim", "الرسم رغم كل الصعاب", "Irises", "İrisler", "سوسن",
         "Wheatfield with a Reaper", "Wheatfield With a Reaper", "Buğday Tarlası ve Orakçı", "حقل قمح مع حاصد",
         "Copy after Millet", "Millet’den Kopya", "نسخة على غرار مييه", "Roses", "Güller",
         # moved from Room 7 on 19 Oct 2026: stop sequences show it is played between
         # 'Painting Against All Odds' and 'Irises' (Saint-Remy = asylum period)
         "Vincent’s Mental Health", "Vincent’ın Ruh Sağlığı", "Психічне здоров'я Вінсента", "صحة فنسنت العقلية"],
 "R10": ["Impassioned Nature", "Tutkulu Doğa", "الطبيعة العاطفية", "Tree Roots", "Ağaç Kökleri", "جذور شجرة",
         "Wheatfield with Crows", "Buğday Tarlası ve Kargalar", "حقل قمح مع غربان", "Landscape at Twilight",
         "Alacakaranlıkta Manzara", "منظر طبيعي عند الغسق",
         # moved from Room 12 on 19 Oct 2026: played between 'Tree Roots' and 'Expression of Colour'
         "Vincent’s Death", "Vincent’ın Ölümü", "وفاة فنسنت"],
 "R11": ["Expression of Colour", "L'espressione del colore", "L'expression de la couleur", "Renklerin Dili",
         "تعابير اللون", "Motherhood", "Maternità", "母亲和孩子", "Maternité", "Maternidad", "Mutterschaft", "Materinstvo",
         "Материнство", "Maternidade", "모성", "母性", "Annelik", "Moederschap", "أمومة", "Maurice Denis, Motherhood",
         "The Blue Dress", "Il vestito blu", "蓝色礼服", "La Robe bleue", "El vestido azul", "Das blaue Kleid",
         "O Vestido Azul", "더 블루 드레스", "Синее платье", "青いドレス", "Mavi Elbise", "De blauwe japon", "Синя сукня",
         "الرداء الأزرق", "Kees van Van Dongen, The Blue Dress",
         "Edvard Munch, Portrait of Félix Auerbach", "Portrait of Félix Auerbach", "弗里克·奥尔巴赫画像",
         "펠릭스 아우어바흐의 초상화", "Edvard Munch, Portret van Félix Auerbach", "Портрет Феликса Ауэрбаха",
         "Félix Auerbach’ın Portresi", "フェリックス・アウアーバッハの肖像", "Портрет Фелікса Ауербаха",
         "لوحة شخصية لفيليكس أورباخ", "The Sower"],
 "R12": ["Final Thoughts", "Son düşünceler", "الأفكار الأخيرة", "Almond Blossom", "盛开的杏花", "Çiçek Açan Badem Ağacı",
         "زهر اللوز"],
 "EXH": [  # Yellow exhibition (13 Feb - 17 May 2026)
         "Welcome", "Introductie", "Setting Sun", "The Yellow Room", "Evening", "Evening, appelé aussi Le Bal",
         "The Yellow Hill", "De Gele Heuvel", "Piles of French Novels", "Three Novels", "Female Head", "Head of a Woman",
         "The Form of Yellow (Notre Dame)", "Large Study", "Grote studie", "Große Studie", "Discolouration", "Décolorés",
         "Descolorido", "褪色", "Verfärbt", "Verkleurd", "Alterazione dei colori", "Der gelbe Klang", "De klank van geel",
         "黄色之声", "Gele pigmenten", "Geelpigmenten", "The Yellow Nineties", "De Hoge Gele Noot", "De kleur van zonlicht",
         "De essentie van geel", "Going To The Ball", "Decadent Young Woman, After the Dance", "Fluwelen avondjapon",
         "Avenue of Poplars in Autumn", "View from Theo's Apartment", "Bank of the Seine", "Shoes",
         # Vincent's Path to Fame (from 12 Jun 2026)
         "Prologue", "Prólogo", "Prolog", "Proloog", "序言", "프롤로그", "ПРОЛОГ", "はじめに", "Introduzione",
         "Benvenuti", "Wilkommen", "ようこそ",
         "Vincent", "文森特", "빈센트", "ВИНСЕНТ", "フィンセント", "فنسنت فان غوخ",
         "Theo", "提奥", "테오", "TEO", "テオ", "Jo", "乔安娜", "조", "ДЖО", "ヨー",
         "Nephew Vincent", "Il nipote Vincent", "El sobrino Vincent", "Le neveu Vincent", "侄子文森特", "Neffe Vincent",
         "Sobrinho Vincent", "조카 빈센트", "甥のフィンセント", "ПЛЕМЯННИК ВИНСЕНТ", "Neefje Vincent",
         "Ingenieur Vincent", "Vincent’s Choice", "La scelta di Vincent", "La decisión de Vincent", "Le choix de Vincent",
         "文森特的选择", "Vincents Entscheidung", "A Escolha de Vincent", "빈센트의 선택", "Vincents keuze",
         "フィンセントの選択", "Выбор Винсента", "Theo's Choice", "Theo’s Choice", "La scelta di Theo",
         "La decisión de Theo", "Le choix de Théo", "提奥的选择", "Theos Entscheidung", "A Escolha de Theo",
         "테오의 선택", "Theo's keuze", "テオの選択", "Выбор Тео", "Jo’s Choice", "La scelta di Jo", "La decisión de Jo",
         "Le choix de Jo", "乔安娜的选择", "Jos Entscheidung", "A Escolha de Jo", "조의 선택", "Jo's keuze", "ヨーの選択",
         "Выбор Джо", "Nephew Vincent’s Choice", "La scelta del nipote Vincent", "La decisión del sobrino Vincent",
         "Le choix du neveu Vincent", "侄子文森特的选择", "Neffe Vincents Entscheidung", "A Escolha do Sobrinho Vincent",
         "조카 빈센트의 선택", "甥のフィンセントの選択", "Выбор племянника Винсента", "Neefje Vincents keuze",
         "Ingenieur Vincents keuze", "Epilogue", "Epílogo", "Epilogo", "Épilogue", "Epilog", "后记", "에필로그",
         "ЭПИЛОГ", "Epiloog"],
}
MEDIUM = {"Roses", "Güller",
          "Welcome", "Introductie", "Benvenuti", "Wilkommen", "ようこそ"}

def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).replace('"', "").split())

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    seen = {}
    rows = []
    for zone, names in STATIONS.items():
        for n in names:
            k = norm(n)
            if k in seen:
                print("DUPLICATE:", k, seen[k], zone)
            seen[k] = zone
            floor, label = ROOMS[zone]
            rows.append({"station_name": k, "zone_id": zone, "floor": floor, "room_label": label,
                         "confidence": "medium" if n in MEDIUM else "high"})
    out = os.path.join(here, "station_room_mapping.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} station names to {out}")
