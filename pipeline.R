

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8db79f9b-c7cd-4f6d-bf6a-5823c5092336"),
    brp_stratify_matches_for_loveplot=Input(rid="ri.foundry.main.dataset.e6de0fd5-5a67-4182-a058-ec3b1eb12fb6")
)
# BRP Match Love Plot (2d456b97-743f-4391-80aa-d37dc626c8bf): v11
brp_love_plot <- function(brp_stratify_matches_for_loveplot) {
    library(cobalt)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(stringr)

    df <- brp_stratify_matches_for_loveplot
    sample_frac <- 0.05
    cohort_col <- "gender_concept_name"
    source_cohort = "MALE"
    target_cohort = "FEMALE"

    # Get list of feature columns
    # categorical columns have suffix _index 
    # continuous columns have suffix _imputed
    df_colnames <- SparkR::colnames(df)
    categorical <- sub('_index', '', df_colnames[grepl("index", df_colnames)])
    nums <- df_colnames[grepl("_imputed", df_colnames)]
    cols <- c(nums, categorical)

    if(sample_frac < 1.0) {
        ##in case the data are too large for R... stratified sample without replacement
        fractions = list()
        fractions[[source_cohort]] <- sample_frac
        fractions[[target_cohort]] <- sample_frac
        df <- SparkR::sampleBy(df, 
                               col = cohort_col, 
                               fractions = fractions, 
                               seed = 42) 
        #print(list(source_cohort = sample_frac, target_cohort = sample_frac))
        #df <- SparkR::sample(df, fraction = sample_frac)
    }

    df_test <- SparkR::collect(SparkR::select(df, cols))
    df_meta <- SparkR::collect(SparkR::select(df, c("treat", "strata")))
    str(df_test)
    str(df_meta)

    # Calculate balance and extract as data frame
    res <- bal.tab(df_test, treat = df_meta$treat, match.strata = df_meta$strata, un = TRUE, disp.v.ratio = TRUE)
    df_post <- res$Balance
    df_post$covariate <- rownames(df_post)

    # calcuate improvement of the matched sample compared to the unmatched
    # and use it to set the orderings of the covariates for plotting
    df_post$improvement <- abs(df_post$Diff.Un) - abs(df_post$Diff.Adj)
    df_post$covariate <- reorder(df_post$covariate, df_post$improvement)

    df_post <- df_post %>%
        # not computed (there's probably a way to not try in the call to bal.tab)
        select(-M.Threshold.Un, -V.Threshold.Un, -M.Threshold, -V.Threshold) %>%
        # we want to plot the values of these variables/columns, so pivot to long
        pivot_longer(c("Diff.Un", "Diff.Adj", "V.Ratio.Un", "V.Ratio.Adj"), names_to = "eval_variable", values_to = "eval_value")

    # rename the variable names and split - e.g. V.Ratio.Un -> Variance Ratio; Unadjusted
    df_post <- df_post %>%
               mutate(eval_variable = str_replace(eval_variable, "V.Ratio", "Variance Ratio")) %>%
               mutate(eval_variable = str_replace(eval_variable, "Diff", "Absolute Standardized Mean Difference")) %>%
               mutate(eval_variable = str_replace(eval_variable, "Un", "Unadjusted")) %>%
               mutate(eval_variable = str_replace(eval_variable, "Adj", "Adjusted")) %>%
               separate(eval_variable, into = c("test", "set"), sep = "\\.")

    # for the mean differences we only care about absolute values for the love plot
    # show absolute difference, direction not needed
    df_post$eval_value[df_post$test == "Absolute Standardized Mean Difference"] <- abs(df_post$eval_value[df_post$test == "Absolute Standardized Mean Difference"])

    p <- ggplot(df_post) +
        geom_point(aes(x = eval_value, y = covariate, color = set)) +
        facet_grid(. ~ test, scales = "free_x", switch = "x") +
        scale_color_discrete(name = "Sample") +
        scale_x_continuous(name = "") +
        scale_y_discrete(name = "Variable") +
        theme_bw() +
        theme(aspect.ratio = 3, axis.text.y = element_text(size = 7)) 

    plot(p)

    return(df_post)
}

