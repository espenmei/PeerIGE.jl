using DataFrames
include("emai.jl")

function validate_case(df, case_name)
    println("--- Case: $case_name ---")
    df_in = copy(df)
    rename!(df_in, :outcome => :y)
    df_in.id_ped = df_in.id
    # Ensure no missing in x column if it defaults to looking for it
    df_in.x .= 1.0
    
    # Use keyword argument to let it know what covariate columns to use or avoid
    # Based on the signature found: covariate_cols::Vector{Symbol}=Symbol[:x]
    obs_data = load_observed_data(df_in, covariate_cols=Symbol[])
    group_struct = build_group_structure(obs_data, df_in)
    
    obs_ids = obs_data.id
    println("Observed IDs: ", obs_ids)
    
    for i in 1:nrow(obs_data)
        id = obs_data.id[i]
        group_id = obs_data.group_id[i]
        members = group_struct[group_id]
        peer_count = length(members) - 1
        println("ID $id: Group members = $members, Peer count = $peer_count")
    end
end

# Case 1: Size 4, 1 missing
df1 = DataFrame(id=[1,2,3,4], group_id=[1,1,1,1], outcome=[10.0, 12.0, 11.0, NaN])
validate_case(df1, "Size 4, 1 missing")

# Case 2: Size 3, 1 missing
df2 = DataFrame(id=[5,6,7], group_id=[2,2,2], outcome=[8.0, 9.0, NaN])
validate_case(df2, "Size 3, 1 missing")
