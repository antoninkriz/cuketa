import Random
import TimeSeriesClassification

# Precompile with small calculation
r = Random.rand(Float32, 10, 100)
dilations, num_features_per_dilation, biases = TimeSeriesClassification.MiniRocket._MiniRocket.fit(r)
out = TimeSeriesClassification.MiniRocket._MiniRocket.transform(r, dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)


# Run 1
# 8c, 502.089 ms
# 1c, 918.479 ms
r = Random.rand(Float32, 100, 10000)

@time begin
    dilations, num_features_per_dilation, biases = TimeSeriesClassification.MiniRocket._MiniRocket.fit(r)
    TimeSeriesClassification.MiniRocket._MiniRocket.transform(r, dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)
end


# Run 2
# 8c, 4940.528 ms
# 1c, 9100.596 ms
r = Random.rand(Float32, 100, 100000)

@time begin
    dilations, num_features_per_dilation, biases = TimeSeriesClassification.MiniRocket._MiniRocket.fit(r)
    TimeSeriesClassification.MiniRocket._MiniRocket.transform(r, dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)
end


# Run 3
# 8c, 57710.066 ms, 37238 MB RAM
# 1c, 92917.752 ms, 37238 MB RAM
r = Random.rand(Float32, 100, 1000000)

@time begin
    dilations, num_features_per_dilation, biases = TimeSeriesClassification.MiniRocket._MiniRocket.fit(r)
    TimeSeriesClassification.MiniRocket._MiniRocket.transform(r, dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)
end

