# BW Navigation - Emergency Routing App

Interactive emergency response routing for emergency management and response planning.

**Features:**
- Drive + walk path optimization
- Helipad candidate visualization
- Real-time route alternatives
- Mobile-friendly interface

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cloud Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud).

### Data

Sample data is included in `/data`:
- `buildings.gpkg` - Building footprints + risk/vulnerability scores
- `vegetation.tif` - Vegetation density raster
- `roads.gpkg` - Road network
- `parks.gpkg` - Parks and candidate LZ locations

To use your own data:
1. Replace files in `/data`
2. Ensure same layer names and coordinate system (EPSG:4326)
3. Commit and push to GitHub
4. Streamlit Cloud will auto-redeploy

## File Structure

```
bw-navigation-app/
├── app.py                    # Main Streamlit entry point
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/                     # GeoPackage and raster files
│   ├── buildings.gpkg
│   ├── vegetation.tif
│   ├── roads.gpkg
│   └── parks.gpkg
└── README.md
```

## Requirements

- Python 3.9+
- GDAL/PROJ (included via rasterio, pyogrio)
- 512MB RAM minimum

## Support

For issues or feature requests, open an issue on GitHub.
