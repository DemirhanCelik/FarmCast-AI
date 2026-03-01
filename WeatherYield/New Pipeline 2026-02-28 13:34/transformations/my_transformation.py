from pyspark import pipelines as dp

@dp.table
def rma_county_yields():
    return spark.read.table("workspace.default.rma_county_yields_report_399")

@dp.table
def noaa_station_month_metrics():
    return spark.read.table("workspace.default.noaa_station_month_metrics_399")

@dp.table(
    name="combined_yields_weather",
    comment="Combined county yields and NOAA metrics on Year and State"
)
def combined_yields_weather():
    yields = rma_county_yields()
    metrics = noaa_station_month_metrics()
    return yields.join(
        metrics,
        (yields["Yield Year"] == metrics["Year"]) &
        (yields["State Name"] == metrics["State Name"]),
        "inner"
    )