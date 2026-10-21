"""Three intro slides on what taste is: (1) kitsch vs a masterpiece, (2) acclaimed-but-looks-bad vs amateur-and-panned,
(3) a painter the critics themselves disagree on. Same 16:9 card style as the findings; report.py puts them first.

  python -m taste.intro   # downloads the images (Wikimedia) into DATA/intro with their licence lines
Facts and quotes are from the English Wikipedia articles on each work (checked 2026-10-21); lay ratings from LAPIS.
"""
import base64, html, io, json, time, urllib.request
import pandas as pd
from PIL import Image
from taste.paths import DATA, RAW

DIR = DATA / "intro"
UA = {"User-Agent": "taste-talk-slides/0.1 (research use)"}
IMAGES = {  # name: (url, licence/credit line shown on the slide)
    "mona_lisa": ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/1920px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
                  "Leonardo da Vinci, public domain (Wikimedia Commons / C2RMF)"),
    "friend_in_need": ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/A_Friend_in_Need_1903_C.M.Coolidge.jpg/1920px-A_Friend_in_Need_1903_C.M.Coolidge.jpg",
                       "C. M. Coolidge, public domain (Wikimedia Commons)"),
    "ecce_homo_restored": ("https://upload.wikimedia.org/wikipedia/en/6/61/Attempted_restoration_of_Ecce_Homo.jpg",
                           "Restoration by Cecilia Giménez; low-resolution image used under fair use (English Wikipedia, via The New York Times)"),
    "ecce_homo_original": ("https://upload.wikimedia.org/wikipedia/commons/b/b7/Elias_Garcia_Martinez_-_Ecce_Homo.jpg",
                           "Elías García Martínez, public domain (Wikimedia Commons)"),
    "birth_of_venus": ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/William-Adolphe_Bouguereau_%281825-1905%29_-_The_Birth_of_Venus_%281879%29.jpg/1920px-William-Adolphe_Bouguereau_%281825-1905%29_-_The_Birth_of_Venus_%281879%29.jpg",
                       "W.-A. Bouguereau, public domain (Wikimedia Commons / Musée d'Orsay)"),
}
LAPIS_PATHS = {"mona_lisa": "figurative/leonardo-da-vinci_mona-lisa.jpg", "lavender_mist": "abstract/jackson-pollock_number-1-lavender-mist(1).jpg"}


def download():
    DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, _) in IMAGES.items():
        p = DIR / f"{name}.jpg"
        if p.exists(): continue
        for attempt in range(5):
            try: p.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()); break
            except urllib.error.HTTPError as e:
                if e.code != 429: raise
                time.sleep(15 * (attempt + 1))
        Image.open(p).verify(); print(p)
        time.sleep(3)
    (DIR / "sources.json").write_text(json.dumps({k: dict(url=u, credit=c) for k, (u, c) in IMAGES.items()}, indent=1))


def uri(path, side=1400):
    im = Image.open(path).convert("RGB"); im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def lapis():
    """LAPIS mean lay rating (0-100) and its percentile among all 10,550 LAPIS artworks, for the works that are in LAPIS."""
    d = pd.concat([pd.read_csv(RAW / f"asmaerten__LAPIS/giaa/{s}.csv") for s in ["train", "test", "val"] if (RAW / f"asmaerten__LAPIS/giaa/{s}.csv").exists()])
    pct = d.mean_response.rank(pct=True) * 100
    return {k: (d.loc[d.path == p, "mean_response"].item(), pct[d.path == p].item(), int(d.loc[d.path == p, "number of raters"].item())) for k, p in LAPIS_PATHS.items()}


def work(img, title, meta, verdicts, credit):
    v = "".join(f'<div class="verdict"><b>{html.escape(k)}</b> {v}</div>' for k, v in verdicts)
    return (f'<figure class="work"><div class="frame">{img}</div><figcaption><div class="wt">{title}</div><div class="wm">{html.escape(meta)}</div>{v}'
            f'<div class="credit">{html.escape(credit)}</div></figcaption></figure>')


def slide(n, title, sub, body, foot):
    return (f'<div class="slide finding intro"><div class="kicker">Intro · {n} of 3</div><h3>{html.escape(title)}</h3><p class="sub">{sub}</p>'
            f'<div class="ibody">{body}</div><p class="foot">{foot}</p></div>')


def lay(k, L):
    m, p, n = L[k]
    rank = f"top {100 - p:.1f}%" if p >= 99 else f"top {100 - p:.0f}%" if p >= 50 else f"bottom {p:.0f}%"
    return f"{m:.0f}/100 from {n} people, {rank} of 10,550 artworks (LAPIS)"


CSS = """
.intro .ibody{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:28px;min-height:0}
.intro .ibody.one{grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr)}.intro .ibody.one .work .frame{height:450px}
.work{margin:0;display:flex;flex-direction:column;gap:10px;min-height:0}
.work .frame{height:340px;flex:none;display:flex;align-items:center;justify-content:center;background:var(--bg);border-radius:10px;padding:8px;position:relative}
.work .frame img{max-width:100%;max-height:100%;object-fit:contain;display:block;border-radius:4px}
.work .frame .inset{position:absolute;left:12px;bottom:12px;width:86px;border:2px solid var(--surface);border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25)}
.work .frame .inset-label{position:absolute;left:12px;bottom:4px;font-size:.66rem;color:var(--ink2);background:var(--surface);padding:0 4px;border-radius:3px}
.work .wt{font-weight:650;font-size:1rem}.work .wm{font-size:.82rem;color:var(--ink2);margin-bottom:4px}
.work .verdict{font-size:.9rem;line-height:1.35;margin-top:2px}.work .verdict b{font-weight:650;margin-right:4px}
.work .credit{font-size:.68rem;color:var(--muted);margin-top:6px}
.timeline{display:flex;flex-direction:column;gap:14px;justify-content:center}
.timeline .era{border-left:3px solid var(--border);padding:2px 0 2px 14px}
.timeline .era .when{font-size:.78rem;font-weight:650;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
.timeline .era .what{font-size:.98rem;line-height:1.4}
.timeline .era.up{border-color:#1baf7a}.timeline .era.down{border-color:#e34948}.timeline .era.split{border-color:#eda100}
@media (max-width:820px){.intro .ibody,.intro .ibody.one{grid-template-columns:1fr}}
"""


def build():
    L = lapis()
    lav = RAW / "asmaerten__LAPIS/images" / LAPIS_PATHS["lavender_mist"]
    img = lambda name: f'<img src="{uri(DIR / f"{name}.jpg")}" alt="">'
    s1 = slide(1, "Some judgments of taste are easy",
               "Nobody needs training to rank these two: critics and lay raters agree.",
               work(img("friend_in_need"), "<i>A Friend in Need</i>, 1903", "Cassius Marcellus Coolidge · oil on canvas",
                    [("Critics:", "kitsch. One of 16 dog paintings commissioned by Brown &amp; Bigelow to advertise cigars; the poker-playing dogs are the stock American example of kitsch in home decoration.")],
                    IMAGES["friend_in_need"][1])
               + work(img("mona_lisa"), "<i>Mona Lisa</i>, c. 1503–1519", "Leonardo da Vinci · oil on poplar · Louvre, Paris",
                      [("Critics:", "a masterpiece of the Italian Renaissance."), ("Lay raters:", lay("mona_lisa", L) + ".")], IMAGES["mona_lisa"][1]),
               "Kitsch characterisation: English Wikipedia, “Dogs Playing Poker”. LAPIS: 10,550 WikiArt paintings, each rated 0–100 by about 20 people.")
    s2 = slide(2, "Some are not: what looks worse can be acclaimed",
               "One looks like spilled paint and hangs in the National Gallery of Art. The other is an earnest figurative effort that critics and the press mocked around the world.",
               work(f'<img src="{uri(lav)}" alt="">', "<i>Number 1, 1950 (Lavender Mist)</i>", "Jackson Pollock · oil, enamel and aluminium paint poured on canvas, 221 × 300 cm",
                    [("Critics:", "a landmark of Abstract Expressionism; titled by Clement Greenberg, in the National Gallery of Art, Washington, since 1976."),
                     ("Lay raters:", lay("lavender_mist", L) + ".")], "Image: WikiArt via the LAPIS dataset")
               + work(img("ecce_homo_restored") + f'<img class="inset" src="{uri(DIR / "ecce_homo_original.jpg", 300)}" alt=""><span class="inset-label">before</span>',
                      "<i>Ecce Homo</i>, after the 2012 restoration", "Fresco by Elías García Martínez, c. 1930 · restored by Cecilia Giménez, an 81-year-old untrained amateur · Borja, Spain",
                      [("Critics and press:", "“a crayon sketch of a very hairy monkey in an ill-fitting tunic” (BBC); nicknamed <i>Ecce Mono</i>."),
                       ("And yet:", "visitors to the church rose from 6,000 to 57,000 a year by 2016.")],
                      IMAGES["ecce_homo_restored"][1] + "; inset: " + IMAGES["ecce_homo_original"][1]),
               "Facts and quotes: English Wikipedia, “Lavender Mist” and “Ecce Homo (García Martínez and Giménez)”.")
    eras = [("split", "In his lifetime (1825–1905)", "To the academic art world, one of the greatest painters alive: Grand Officier of the Legion of Honour, shown at every Paris Salon of his career. "
             "To Degas's circle, <i>Bouguereauté</i> was an insult; the critic Richard Muther (1894) found him “destitute of artistic feeling but possessing a cultured taste”."),
            ("down", "After 1920", "Fell into disrepute as tastes changed; Kenneth Clark faulted his “convention of smoothed-out form and waxen surface”."),
            ("up", "Since the 1980s", "Rediscovered with a revival of interest in figure painting: retrospectives in Montreal (1984) and Milwaukee (2019).")]
    timeline = '<div class="timeline">' + "".join(f'<div class="era {c}"><div class="when">{w}</div><div class="what">{t}</div></div>' for c, w, t in eras) + "</div>"
    s3 = slide(3, "And sometimes even the critics disagree",
               "Critics split over Bouguereau while he lived, wrote him off after his death, and have been revising their verdict since.",
               work(img("birth_of_venus"), "<i>The Birth of Venus</i>, 1879", "William-Adolphe Bouguereau · oil on canvas · Musée d'Orsay, Paris", [], IMAGES["birth_of_venus"][1]) + timeline,
               "Facts and quotes: English Wikipedia, “William-Adolphe Bouguereau”.").replace('class="ibody"', 'class="ibody one"')
    return [s1, s2, s3], CSS


if __name__ == "__main__":
    download()
