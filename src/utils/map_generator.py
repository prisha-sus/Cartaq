import h3
import folium
import os

def hex_map_html(h3_index: str) -> str:
    try:
        hex_center = h3.cell_to_latlng(h3_index)
        hex_boundary = h3.cell_to_boundary(h3_index)
    except AttributeError:
        hex_center = h3.h3_to_geo(h3_index)
        hex_boundary = h3.h3_to_geo_boundary(h3_index)

    m = folium.Map(location=hex_center, zoom_start=15, tiles="CartoDB positron")

    folium.Polygon(
        locations=hex_boundary,
        color="#e74c3c", # Alert Red
        weight=2,
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.4,
        tooltip=f"Target Intervention Zone: {h3_index}"
    ).add_to(m)

    map_html = m.get_root().render()
    return map_html, m

if __name__ == "__main__":
    test_hex = "8860a25997fffff"
    html_string, folium_map_obj = hex_map_html(test_hex)

    os.makedirs("output", exist_ok=True)
    folium_map_obj.save("output/test_map.html")

    print("--- Map Generation Complete ---")
    print(f"Generated raw HTML string of length: {len(html_string)}")