suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(geepack))
suppressPackageStartupMessages(library(rms))
suppressPackageStartupMessages(library(pROC))
suppressPackageStartupMessages(library(MatchIt))
suppressPackageStartupMessages(library(sandwich))
suppressPackageStartupMessages(library(lmtest))
suppressPackageStartupMessages(library(splines))
suppressPackageStartupMessages(library(boot))
suppressPackageStartupMessages(library(jsonlite))

read_cohorts <- function(path) {
  value <- read.csv(path, stringsAsFactors = FALSE)
  value$group_binary <- as.integer(value$group == "implementation")
  value$events <- round(value$n * value$adr_percent / 100)
  value$non_events <- value$n - value$events
  value
}

read_outcomes <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

read_temporal <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

read_moderators <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

read_sensitivity <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

validate_cohorts <- function(data) {
  stopifnot(nrow(data) == 16)
  stopifnot(sum(data$group == "implementation") == 8)
  stopifnot(sum(data$group == "unsupported") == 8)
  stopifnot(all(data$n > 0))
  stopifnot(all(data$adr_percent > 0))
  stopifnot(all(data$adr_percent < 100))
  stopifnot(all(data$fidelity_score[data$group == "implementation"] >= 0))
  stopifnot(all(data$fidelity_score[data$group == "implementation"] <= 100))
  invisible(data)
}

aggregate_group <- function(data, group_name) {
  selected <- data[data$group == group_name, ]
  events <- sum(selected$events)
  total <- sum(selected$n)
  list(
    group = group_name,
    events = events,
    total = total,
    rate = events / total,
    percent = 100 * events / total
  )
}

wilson_interval <- function(events, total, confidence = 0.95) {
  probability <- events / total
  critical <- qnorm(0.5 + confidence / 2)
  denominator <- 1 + critical^2 / total
  center <- (probability + critical^2 / (2 * total)) / denominator
  margin <- critical * sqrt(probability * (1 - probability) / total + critical^2 / (4 * total^2)) / denominator
  c(lower = center - margin, upper = center + margin)
}

newcombe_interval <- function(first_events, first_total, second_events, second_total, confidence = 0.95) {
  first <- first_events / first_total
  second <- second_events / second_total
  first_ci <- wilson_interval(first_events, first_total, confidence)
  second_ci <- wilson_interval(second_events, second_total, confidence)
  lower <- first - second - sqrt((first - first_ci[["lower"]])^2 + (second_ci[["upper"]] - second)^2)
  upper <- first - second + sqrt((first_ci[["upper"]] - first)^2 + (second - second_ci[["lower"]])^2)
  c(lower = lower, upper = upper)
}

odds_ratio_interval <- function(a, b, c, d, confidence = 0.95) {
  cells <- c(a, b, c, d)
  if (any(cells == 0)) {
    cells <- cells + 0.5
  }
  log_ratio <- log(cells[[1]] * cells[[4]] / (cells[[2]] * cells[[3]]))
  standard_error <- sqrt(sum(1 / cells))
  critical <- qnorm(0.5 + confidence / 2)
  c(
    estimate = exp(log_ratio),
    lower = exp(log_ratio - critical * standard_error),
    upper = exp(log_ratio + critical * standard_error),
    standard_error = standard_error
  )
}

pooled_comparison <- function(data, confidence = 0.95) {
  implementation <- aggregate_group(data, "implementation")
  unsupported <- aggregate_group(data, "unsupported")
  difference <- implementation$rate - unsupported$rate
  difference_ci <- newcombe_interval(
    implementation$events,
    implementation$total,
    unsupported$events,
    unsupported$total,
    confidence
  )
  ratio <- odds_ratio_interval(
    implementation$events,
    implementation$total - implementation$events,
    unsupported$events,
    unsupported$total - unsupported$events,
    confidence
  )
  contingency <- matrix(
    c(
      implementation$events,
      implementation$total - implementation$events,
      unsupported$events,
      unsupported$total - unsupported$events
    ),
    nrow = 2,
    byrow = TRUE
  )
  fisher <- fisher.test(contingency)
  nnt <- 1 / abs(difference)
  cohen_h <- 2 * asin(sqrt(implementation$rate)) - 2 * asin(sqrt(unsupported$rate))
  list(
    implementation = implementation,
    unsupported = unsupported,
    difference_pp = 100 * difference,
    difference_lower_pp = 100 * difference_ci[["lower"]],
    difference_upper_pp = 100 * difference_ci[["upper"]],
    odds_ratio = ratio[["estimate"]],
    odds_ratio_lower = ratio[["lower"]],
    odds_ratio_upper = ratio[["upper"]],
    p_value = fisher$p.value,
    nnt = nnt,
    cohen_h = cohen_h
  )
}

cohort_log_odds <- function(events, non_events, correction = 0.5) {
  if (events == 0 || non_events == 0) {
    events <- events + correction
    non_events <- non_events + correction
  }
  c(estimate = log(events / non_events), variance = 1 / events + 1 / non_events)
}

cohort_effects <- function(data) {
  values <- t(mapply(cohort_log_odds, data$events, data$non_events))
  data.frame(
    cohort_id = data$cohort_id,
    group = data$group,
    estimate = values[, "estimate"],
    variance = values[, "variance"],
    stringsAsFactors = FALSE
  )
}

heterogeneity <- function(estimates, variances) {
  weights <- 1 / variances
  center <- sum(weights * estimates) / sum(weights)
  q <- sum(weights * (estimates - center)^2)
  degrees <- length(estimates) - 1
  denominator <- sum(weights) - sum(weights^2) / sum(weights)
  tau_squared <- max(0, (q - degrees) / denominator)
  i_squared <- ifelse(q > 0, max(0, (q - degrees) / q) * 100, 0)
  list(
    q = q,
    degrees_freedom = degrees,
    p_value = pchisq(q, degrees, lower.tail = FALSE),
    tau_squared = tau_squared,
    i_squared = i_squared
  )
}

group_heterogeneity <- function(data, group_name) {
  effects <- cohort_effects(data[data$group == group_name, ])
  heterogeneity(effects$estimate, effects$variance)
}

expand_cohort <- function(row) {
  data.frame(
    outcome = c(rep(1, row$events), rep(0, row$non_events)),
    group = row$group_binary,
    cohort = row$cohort_id,
    fidelity = ifelse(is.na(row$fidelity_score), 0, row$fidelity_score),
    stringsAsFactors = FALSE
  )
}

expand_cohorts <- function(data) {
  frames <- lapply(seq_len(nrow(data)), function(index) expand_cohort(data[index, ]))
  do.call(rbind, frames)
}

aggregate_logistic <- function(data) {
  model <- glm(
    cbind(events, non_events) ~ group_binary,
    data = data,
    family = binomial(link = "logit")
  )
  coefficients <- summary(model)$coefficients
  interval <- confint.default(model)
  list(
    model = model,
    coefficients = coefficients,
    interval = interval,
    odds_ratio = exp(coef(model)[["group_binary"]]),
    odds_ratio_lower = exp(interval["group_binary", 1]),
    odds_ratio_upper = exp(interval["group_binary", 2]),
    aic = AIC(model),
    bic = BIC(model)
  )
}

gee_exchangeable <- function(data) {
  expanded <- expand_cohorts(data)
  model <- geeglm(
    outcome ~ group,
    id = cohort,
    data = expanded,
    family = binomial(link = "logit"),
    corstr = "exchangeable"
  )
  coefficients <- summary(model)$coefficients
  estimate <- coefficients["group", "Estimate"]
  standard_error <- coefficients["group", "Std.err"]
  critical <- qnorm(0.975)
  list(
    model = model,
    coefficients = coefficients,
    odds_ratio = exp(estimate),
    odds_ratio_lower = exp(estimate - critical * standard_error),
    odds_ratio_upper = exp(estimate + critical * standard_error),
    p_value = coefficients["group", "Pr(>|W|)"]
  )
}

fidelity_data <- function(data) {
  selected <- data[data$group == "implementation", ]
  selected$improvement <- c(8.0, 7.1, 8.4, 3.0, 5.1, 2.8, 7.5, 4.5)
  selected
}

fidelity_correlation <- function(data) {
  selected <- fidelity_data(data)
  test <- cor.test(
    selected$fidelity_score,
    selected$improvement,
    method = "spearman",
    exact = FALSE
  )
  list(
    coefficient = unname(test$estimate),
    p_value = test$p.value,
    n = nrow(selected)
  )
}

restricted_spline_model <- function(data) {
  selected <- fidelity_data(data)
  knots <- quantile(selected$fidelity_score, c(0.25, 0.50, 0.75))
  model <- lm(
    improvement ~ rcs(fidelity_score, knots),
    data = selected
  )
  grid <- data.frame(
    fidelity_score = seq(
      min(selected$fidelity_score),
      max(selected$fidelity_score),
      length.out = 1001
    )
  )
  prediction <- predict(model, newdata = grid, interval = "confidence")
  grid$prediction <- prediction[, "fit"]
  grid$lower <- prediction[, "lwr"]
  grid$upper <- prediction[, "upr"]
  half_maximum <- min(grid$prediction) + 0.5 * diff(range(grid$prediction))
  ed50_index <- which.min(abs(grid$prediction - half_maximum))
  effective_indices <- which(grid$lower > 0)
  slopes <- c(diff(grid$prediction) / diff(grid$fidelity_score), NA) * 10
  plateau_indices <- which(slopes < 0.5)
  list(
    model = model,
    knots = knots,
    grid = grid,
    ed50 = grid$fidelity_score[[ed50_index]],
    minimum_effective = ifelse(length(effective_indices) > 0, grid$fidelity_score[[effective_indices[[1]]]], NA),
    plateau = ifelse(length(plateau_indices) > 0, grid$fidelity_score[[plateau_indices[[1]]]], NA),
    slopes = slopes
  )
}

optimal_fidelity_threshold <- function(data, meaningful = 5) {
  selected <- fidelity_data(data)
  labels <- selected$improvement >= meaningful
  receiver <- roc(labels, selected$fidelity_score, quiet = TRUE, direction = "<")
  coordinates <- coords(
    receiver,
    x = "best",
    best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity")
  )
  list(
    threshold = as.numeric(coordinates$threshold),
    sensitivity = as.numeric(coordinates$sensitivity),
    specificity = as.numeric(coordinates$specificity),
    auroc = as.numeric(auc(receiver))
  )
}

bootstrap_threshold <- function(data, repetitions = 200000, seed = 2025) {
  set.seed(seed)
  selected <- fidelity_data(data)
  thresholds <- numeric(repetitions)
  valid <- logical(repetitions)
  for (index in seq_len(repetitions)) {
    sampled_indices <- sample(seq_len(nrow(selected)), replace = TRUE)
    sampled <- selected[sampled_indices, ]
    labels <- sampled$improvement >= 5
    if (length(unique(labels)) < 2) {
      next
    }
    receiver <- roc(labels, sampled$fidelity_score, quiet = TRUE, direction = "<")
    value <- coords(receiver, x = "best", best.method = "youden", ret = "threshold")
    thresholds[[index]] <- as.numeric(value)
    valid[[index]] <- TRUE
  }
  values <- thresholds[valid]
  list(
    estimate = median(values),
    lower = quantile(values, 0.025),
    upper = quantile(values, 0.975),
    valid_repetitions = length(values)
  )
}

paper_decomposition <- function() {
  effects <- c(
    performance_feedback = 2.1,
    detection = 1.5,
    alert_management = 1.4,
    training = 0.8
  )
  shares <- 100 * effects / sum(effects)
  data.frame(
    component = names(effects),
    effect_pp = unname(effects),
    share_percent = unname(shares),
    stringsAsFactors = FALSE
  )
}

gap_recovery <- function(improvement, benchmark = 8.1) {
  100 * improvement / benchmark
}

residual_gap <- function(improvement, benchmark = 8.1) {
  benchmark - improvement
}

e_value <- function(risk_ratio) {
  value <- ifelse(risk_ratio >= 1, risk_ratio, 1 / risk_ratio)
  value + sqrt(value * (value - 1))
}

e_value_odds_ratio <- function(odds_ratio, common_outcome = TRUE) {
  value <- ifelse(common_outcome, sqrt(ifelse(odds_ratio >= 1, odds_ratio, 1 / odds_ratio)), odds_ratio)
  e_value(value)
}

temporal_analysis <- function(data) {
  selected <- data[complete.cases(data[, c("difference", "alert_to_action")]), ]
  time <- seq_len(nrow(selected))
  effect_model <- lm(difference ~ time, data = selected)
  alert_model <- lm(alert_to_action ~ time, data = selected)
  correlation <- cor.test(selected$difference, selected$alert_to_action, method = "pearson")
  initial <- selected$difference[[1]]
  final <- selected$difference[[nrow(selected)]]
  list(
    initial = initial,
    final = final,
    absolute_change = initial - final,
    relative_attenuation = 100 * (initial - final) / initial,
    retained_fraction = 100 * final / initial,
    effect_slope = coef(effect_model)[["time"]],
    alert_slope = coef(alert_model)[["time"]],
    effect_alert_correlation = unname(correlation$estimate),
    effect_alert_p_value = correlation$p.value
  )
}

leave_one_out <- function(data) {
  records <- lapply(seq_len(nrow(data)), function(index) {
    retained <- data[-index, ]
    comparison <- pooled_comparison(retained)
    data.frame(
      omitted_cohort = data$cohort_id[[index]],
      difference_pp = comparison$difference_pp,
      odds_ratio = comparison$odds_ratio,
      nnt = comparison$nnt
    )
  })
  do.call(rbind, records)
}

high_fidelity_analysis <- function(data, cutoff = 80) {
  implementation <- data[data$group == "implementation" & data$fidelity_score >= cutoff, ]
  unsupported <- data[data$group == "unsupported", ]
  unsupported <- unsupported[seq_len(min(nrow(implementation), nrow(unsupported))), ]
  pooled_comparison(rbind(implementation, unsupported))
}

volume_stratified <- function(data) {
  tiers <- unique(data$volume_tier)
  results <- lapply(tiers, function(tier) {
    selected <- data[data$volume_tier == tier, ]
    comparison <- pooled_comparison(selected)
    data.frame(
      volume_tier = tier,
      difference_pp = comparison$difference_pp,
      odds_ratio = comparison$odds_ratio,
      p_value = comparison$p_value
    )
  })
  do.call(rbind, results)
}

paper_targets <- function() {
  list(
    adr_difference_pp = 5.8,
    adjusted_odds_ratio = 1.28,
    adjusted_odds_ratio_lower = 1.11,
    adjusted_odds_ratio_upper = 1.47,
    p_value = 0.001,
    nnt = 17,
    tau_squared = 0.013,
    i_squared = 31,
    spearman = 0.71,
    spearman_p = 0.004,
    ed50 = 68,
    minimum_effective = 58,
    plateau = 85,
    optimal_threshold = 72,
    auroc = 0.89,
    sensitivity = 0.86,
    specificity = 0.83
  )
}

reaim_profile <- function() {
  list(
    reach = list(activation_rate = 89.2, patient_coverage = 97.4),
    effectiveness = list(adr_improvement = 5.8, apc_improvement = 0.22),
    adoption = list(provider_adoption = 93.7, regular_use = 84.3),
    implementation = list(composite_fidelity = 78.4, training = 100, feedback = 71.2, alert = 64.1),
    maintenance = list(month_12 = 4.5, month_18 = 3.8)
  )
}

run_analysis <- function(root, output) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  cohorts <- read_cohorts(file.path(root, "data", "cohorts.csv"))
  temporal <- read_temporal(file.path(root, "data", "temporal.csv"))
  validate_cohorts(cohorts)
  comparison <- pooled_comparison(cohorts)
  logistic <- aggregate_logistic(cohorts)
  gee <- gee_exchangeable(cohorts)
  spline <- restricted_spline_model(cohorts)
  threshold <- optimal_fidelity_threshold(cohorts)
  correlation <- fidelity_correlation(cohorts)
  results <- list(
    primary = comparison,
    logistic = logistic[names(logistic) != "model"],
    gee = gee[names(gee) != "model"],
    implementation_heterogeneity = group_heterogeneity(cohorts, "implementation"),
    unsupported_heterogeneity = group_heterogeneity(cohorts, "unsupported"),
    fidelity_correlation = correlation,
    spline = list(
      knots = spline$knots,
      ed50 = spline$ed50,
      minimum_effective = spline$minimum_effective,
      plateau = spline$plateau
    ),
    threshold = threshold,
    decomposition = paper_decomposition(),
    temporal = temporal_analysis(temporal),
    gap_recovery = gap_recovery(5.8),
    residual_gap = residual_gap(5.8),
    e_value = e_value_odds_ratio(1.28),
    reaim = reaim_profile(),
    targets = paper_targets()
  )
  write.csv(leave_one_out(cohorts), file.path(output, "r_leave_one_out.csv"), row.names = FALSE)
  write.csv(spline$grid, file.path(output, "r_spline_curve.csv"), row.names = FALSE)
  write.csv(volume_stratified(cohorts), file.path(output, "r_volume_stratified.csv"), row.names = FALSE)
  write_json(results, file.path(output, "r_analysis.json"), pretty = TRUE, auto_unbox = TRUE, null = "null")
  invisible(results)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) >= 2) {
  run_analysis(arguments[[1]], arguments[[2]])
}
