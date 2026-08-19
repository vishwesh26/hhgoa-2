import urllib.request
from pathlib import Path

SCREENS = [
    {
        "id": "d7c8c4bff2354bdaa9552eb67fbb1a60",
        "name": "documentation",
        "title": "Documentation",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ8Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpbCiVodG1sXzAwMDY1OTY0MGYyNjAyY2YwMjhmMDdiOTE2MTVjZjdhEgsSBxDj6_HMmA4YAZIBJAoKcHJvamVjdF9pZBIWQhQxNjYxNzAyMDc3MjMyMzkyMjkwNw&filename=&opi=89354086",
        "img_url": "https://lh3.googleusercontent.com/aida/AP1WRLuv-rox-BsELklIuS2PYJUQZyiZwyVo90umn5TfRwMEhqgUiTStVMG_RN1hrpDADo65htGqflb_ysKdu9c1xchDk2R_xZMxUFFJgZ6a21WVfRhxcKitioTy9fn71rBqLCJLFJ-vTE75rjORH-xmihPrCigljee9t5Rpk5TUDCxFJiviXHNhkev_NVIjF-_KEDhxWwT_Eg0Kip8qlKIzLJWoj-olsj4PZeUknByeqRZrUXhu7iNvdgfKlIzG"
    },
    {
        "id": "ee3e036e40fb46578c4c1e8a81ea6a0c",
        "name": "performance_analytics",
        "title": "Performance Analytics",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ8Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpbCiVodG1sXzAwMDY1OTY0MGYzMTRmYzAwMzkyY2FkZjdjMGIwOGYxEgsSBxDj6_HMmA4YAZIBJAoKcHJvamVjdF9pZBIWQhQxNjYxNzAyMDc3MjMyMzkyMjkwNw&filename=&opi=89354086",
        "img_url": "https://lh3.googleusercontent.com/aida/AP1WRLvcAW2XFzGNAExQHM80Ryanf6iBKlZrQ51UtTZmJBbcq82-nqyQyjQ__uCWRyZeueaEifQSDeFdA8aSd2O1eKm7AsVZOzMtOd_PRoti5-p6_dtxdeQeYg2dMdvzTRtQT2pJu87tF-9oMsyS7ybCpHTomRv5_kUvUszZriekvBpbwciVhH6tXde4xwDJA155898h-w8dck5OozoDE0mD8bu-6rohjF-Iz_rAXI3TTDQjQ8ecXn7DeLSNW74H"
    },
    {
        "id": "837c3735f192477fbba0342f6927aa41",
        "name": "kinetic_dashboard",
        "title": "Kinetic Dashboard",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ8Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpbCiVodG1sXzAwMDY1OTY0MjIzYjIzNmMwMjhmMDkxYzUwMDJkMDUzEgsSBxDj6_HMmA4YAZIBJAoKcHJvamVjdF9pZBIWQhQxNjYxNzAyMDc3MjMyMzkyMjkwNw&filename=&opi=89354086",
        "img_url": "https://lh3.googleusercontent.com/aida/AP1WRLubNfu10GSzK5lPQAOAtSTEruSvWgVBhiM0Vlyj8KvbCofaGvN0ZaAf7T8Okovv1Qf1xwQQ00L1O-z9O6YwjokiG7mokwJ97tvbYEDQ-hQe87eyChx1657tlcDisVD4bndE3NJbnQ8wkMxhUMZcKac0mwpAhFj4tW3qkMDdhWkwHmfrAqGaHj8t_nIZ0j1G33x0aoEF54C_6mxIF21WHQPNor3Ovquyd-_0fddBGacTYi9SR2ckLudAFg26"
    },
    {
        "id": "3b29c3a7cf57418492691671e60ad87f",
        "name": "profile_settings",
        "title": "Profile & Settings",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ8Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpbCiVodG1sXzAwMDY1OTY0MGYzZTU1YWUwN2M0Y2YwZDA3MmZhZjhmEgsSBxDj6_HMmA4YAZIBJAoKcHJvamVjdF9pZBIWQhQxNjYxNzAyMDc3MjMyMzkyMjkwNw&filename=&opi=89354086",
        "img_url": "https://lh3.googleusercontent.com/aida/AEtjO1XImPBMIncLn-3OTpEJewkZCMWCpIFOb2U3mjI_EjP_orV5X-dudsTPxn4W5uTLf8q6CaxXETnhRGlLtu2dv9GWxDRmu5NtgyKAb0WRvrXlNJgq-zg78OBDmTAEUM4zDbdRvVlD11E0u6UQjl1PFClOuxWc7ggfAszSphaOnJNWs1OOYYb34yjtHFN-wJ66hlIIrtEqZu3SkVYjNEACYjJqTlTFSW1EGINHONs6bIwgn_gpPusEravjqj_7"
    }
]

out_dir = Path("./stitch_screens")
out_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

for s in SCREENS:
    html_file = out_dir / f"{s['name']}.html"
    img_file = out_dir / f"{s['name']}.png"
    
    print(f"Downloading {s['title']} HTML...")
    req = urllib.request.Request(s["html_url"], headers=headers)
    with urllib.request.urlopen(req) as resp, open(html_file, "wb") as f:
        f.write(resp.read())
        
    print(f"Downloading {s['title']} Screenshot...")
    req_img = urllib.request.Request(s["img_url"], headers=headers)
    with urllib.request.urlopen(req_img) as resp, open(img_file, "wb") as f:
        f.write(resp.read())
        
print("All 4 Stitch screens and screenshots downloaded successfully!")
