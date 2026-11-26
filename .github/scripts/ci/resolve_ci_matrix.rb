#!/usr/bin/env ruby
# frozen_string_literal: true
# Resolve a deterministic FlagCX CI plan. Unknown paths use the full matrix;
# planning never blocks executable work.

require "json"
require "yaml"
require "digest"

config_dir = ARGV.fetch(0) { abort "Usage: #{$PROGRAM_NAME} <config-dir> <context.json>" }
context_path = ARGV.fetch(1) { abort "Usage: #{$PROGRAM_NAME} <config-dir> <context.json>" }
context = JSON.parse(File.read(context_path))

load_yaml = lambda do |path|
  contents = File.read(path)
  begin
    YAML.safe_load(contents, permitted_classes: [], permitted_symbols: [], aliases: false, filename: path) || {}
  rescue ArgumentError
    YAML.safe_load(contents, [], [], false, path) || {}
  end
end

registry_path = File.join(config_dir, "platforms.yml")
abort "Platform registry not found: #{registry_path}" unless File.file?(registry_path)
registry = load_yaml.call(registry_path)
settings = registry.fetch("platforms", {})
abort "#{registry_path}: platforms must be a map" unless settings.is_a?(Hash)

platforms = settings.map do |name, value|
  next if value.is_a?(Hash) && value.fetch("enabled", true) == false
  config_path = File.join(config_dir, "#{name}.yml")
  abort "Platform config not found: #{config_path}" unless File.file?(config_path)
  config = load_yaml.call(config_path)
  abort "#{config_path}: hardware_name must be #{name}" unless config["hardware_name"] == name
  patterns = config.fetch("paths_watch", [])
  abort "#{config_path}: paths_watch must be a non-empty array" unless patterns.is_a?(Array) && !patterns.empty?
  ignore_test = config.fetch("ignore_test", {})
  abort "#{config_path}: ignore_test must be a map" unless ignore_test.is_a?(Hash)
  {"platform" => name, "display_name" => config.fetch("display_name"),
   "patterns" => patterns, "ignore_test" => ignore_test}
end.compact
platform_names = platforms.map { |entry| entry.fetch("platform") }
abort "No enabled platform configs found in #{config_dir}" if platforms.empty?
suites_config_path = File.join(config_dir, "suites.yml")
suites_config = File.file?(suites_config_path) ? load_yaml.call(suites_config_path) : {}

canonical_platform = ->(name) { name == "musa" ? "mthreads" : name }
valid_levels = %w[L0 L1 L2 L3]
normalise_level = lambda do |value|
  level = value.to_s.upcase
  abort "Unknown CI test level: #{value}" unless valid_levels.include?(level)
  level
end
level_number = ->(value) { value.delete_prefix("L").to_i }

labels = Array(context["labels"]).map(&:to_s).map do |label|
  if label.start_with?("platform:")
    names = label.delete_prefix("platform:").split(",").map { |name| canonical_platform.call(name.strip) }
    "platform:#{names.join(',')}"
  else
    label
  end
end
files = Array(context["files"]).map(&:to_s)
requested_suites = Array(context["suites"]).map(&:to_s)
labels.each do |label|
  requested_suites.concat(label.delete_prefix("suite:").split(",")) if label.start_with?("suite:")
end
requested_suites = requested_suites.map(&:strip).reject { |suite| suite.empty? || suite == "all" }.uniq
event_name = context.fetch("event_name")
selection = nil
active = true
routing_reason = []

match_paths = lambda do |patterns, paths|
  patterns.any? do |pattern|
    paths.any? { |path| File.fnmatch?(pattern, path, File::FNM_PATHNAME | File::FNM_EXTGLOB) }
  end
end
select_paths = lambda do
  shared_patterns = registry.fetch("shared_paths_watch", [])
  abort "#{registry_path}: shared_paths_watch must be an array" unless shared_patterns.is_a?(Array)
  if files.empty?
    routing_reason << "No changed files selected no hardware platforms"
    next []
  end
  if match_paths.call(shared_patterns, files)
    routing_reason << "Shared path selected the full platform matrix"
    next platform_names
  end
  selected_names = []
  files.each do |path|
    affected = platforms.filter_map do |entry|
      entry.fetch("platform") if match_paths.call(entry.fetch("patterns"), [path])
    end
    if affected.empty?
      routing_reason << "Unmapped path #{path} selected the full platform matrix"
      selected_names = platform_names
      break
    end
    selected_names.concat(affected)
    routing_reason << "#{path} matched #{affected.join(", ")}"
  end
  selected_names.uniq
end
parse_selection = lambda do |value|
  requested = value.to_s.split(",").map(&:strip).reject(&:empty?).map(&canonical_platform)
  requested = ["all"] if requested.empty?
  unknown = requested.reject { |name| name == "all" || platform_names.include?(name) }
  abort "Unknown platform selection(s): #{unknown.join(", ")}" unless unknown.empty?
  requested.include?("all") ? platform_names : requested.uniq
end

level = normalise_level.call(context.fetch("level", "L2"))
level_labels = labels.filter_map { |label| label.delete_prefix("level:") if label.start_with?("level:") }
level = normalise_level.call(level_labels.max_by { |value| level_number.call(value) }) unless level_labels.empty?

case event_name
when "pull_request"
  route_labels = labels.select { |label| label.start_with?("platform:") }
  requested_platforms = route_labels.flat_map { |label| label.delete_prefix("platform:").split(",") }
  invalid_labels = requested_platforms
                                .reject { |name| name == "all" || platform_names.include?(name) }
  abort "Unknown platform label(s): #{invalid_labels.join(", ")}" unless invalid_labels.empty?
  if requested_platforms.include?("all")
    selection = platform_names
    routing_reason << "Platform label included all and selected the full platform matrix"
  elsif !route_labels.empty?
    selection = requested_platforms.uniq
    routing_reason << "Platform labels selected #{selection.join(", ")}"
  else
    selection = select_paths.call
  end
when "issue_comment"
  permission = context.fetch("permission", "none")
  comment = context.fetch("comment", "").to_s.strip
  command = comment.match(%r{\A/test(?:\s+(all|[a-z0-9_-]+))?\z})
  if context["is_pull_request"] != true || !%w[admin write maintain].include?(permission) || command.nil?
    active = false
    selection = []
    routing_reason << "No authorized /test request; hardware execution is inactive"
  elsif command[1].nil?
    route_labels = labels.select { |label| label.start_with?("platform:") }
    selection = route_labels.empty? ? select_paths.call : parse_selection.call(route_labels.flat_map { |label| label.delete_prefix("platform:").split(",") }.join(","))
    routing_reason << "Authorized /test request used PR routing"
  else
    selection = parse_selection.call(command[1])
    routing_reason << "Authorized /test request selected #{selection.join(", ")}"
  end
when "workflow_dispatch"
  selection = parse_selection.call(context.fetch("selection", "all"))
  routing_reason << "Manual dispatch selected #{selection.join(", ")}"
when "push"
  selection = select_paths.call
when "schedule"
  selection = platform_names
  level = "L3"
  routing_reason << "Scheduled nightly run uses the complete configured matrix at L3"
else
  abort "Unsupported event: #{event_name}"
end

explicit_platform_selection = labels.any? { |label| label.start_with?("platform:") } ||
                              (event_name == "workflow_dispatch" && context.fetch("selection", "all") != "all") ||
                              (event_name == "issue_comment" && context.fetch("comment", "").match?(/\A\/test\s+(?!all\z)[a-z0-9_-]+\z/))
if active && level == "L3" && !explicit_platform_selection && selection != platform_names
  selection = platform_names
  routing_reason << "level:L3 selected the full platform matrix"
end

selected = platforms.select { |entry| selection.include?(entry.fetch("platform")) }
ignore_entries = selected.flat_map do |entry|
  entry.fetch("ignore_test").flat_map do |section, value|
    (value.is_a?(Array) ? value : [value]).filter_map do |item|
      next unless item.is_a?(Hash)
      name = item["name"] || item["test"] || item["phase"] || item["suite"] || section
      {"platform" => entry.fetch("platform"), "kind" => section}.merge(item).merge(
        "name" => name,
        "reason" => item["reason"] || "configured platform exclusion"
      )
    end
  end
end
plan_status = ignore_entries.empty? ? "ready" : "partial"
unit_suites = selected.to_h do |entry|
  config = load_yaml.call(File.join(config_dir, "#{entry.fetch("platform")}.yml"))
  [entry.fetch("platform"), config.fetch("unit_test_suites", [])]
end
unsupported_suites = requested_suites - unit_suites.values.flatten.uniq
abort "Selected platforms do not provide requested suite(s): #{unsupported_suites.join(", ")}" unless unsupported_suites.empty?
suite_shared_paths = suites_config.fetch("shared_paths_watch", [])
suite_path_map = suites_config.fetch("suites", {})
matrix_entries = selected.map do |entry|
  platform = entry.fetch("platform")
  available = unit_suites.fetch(platform)
  if requested_suites.any?
    suites = requested_suites & available
  elsif event_name == "schedule" || match_paths.call(suite_shared_paths, files) || files.empty?
    suites = available
  else
    matched = available.select { |suite| match_paths.call(Array(suite_path_map[suite]), files) }
    unmapped = files.any? do |path|
      !suite_shared_paths.any? { |pattern| File.fnmatch?(pattern, path, File::FNM_PATHNAME | File::FNM_EXTGLOB) } &&
        !suite_path_map.values.flatten.any? { |pattern| File.fnmatch?(pattern, path, File::FNM_PATHNAME | File::FNM_EXTGLOB) }
    end
    suites = unmapped || matched.empty? ? available : matched
  end
  next if suites.empty?
  { "platform" => platform, "display_name" => entry.fetch("display_name"), "suites" => suites }
end.compact
matrix = { "include" => matrix_entries }
suite_filter_active = matrix_entries.any? do |entry|
  entry.fetch("suites").length < unit_suites.fetch(entry.fetch("platform")).length
end
suite_filter_active ||= requested_suites.any?
planned_groups = []
planned_groups << "unit suites" unless level == "L0"
planned_groups.concat(["performance", "Torch API"]) if %w[L2 L3].include?(level)
plan = {
  "schema_version" => 1,
  "source_sha" => context.fetch("ref", ""),
  "event" => event_name,
  "level" => level,
  "status" => plan_status,
  "blocked" => [],
  "platforms" => selected.map { |entry| entry.fetch("platform") },
  "unit_suites" => unit_suites,
  "planned_groups" => planned_groups,
  "routing_reason" => routing_reason,
  "changed_files" => files,
  "ignore_test" => ignore_entries,
  "gate_policy" => {
    "blocked_is_non_fatal_when_not_required" => true,
    "ignore_test_is_visible_not_passed" => true
  }
}
plan["plan_hash"] = Digest::SHA256.hexdigest(JSON.generate(plan))

plan_path = ENV["FLAGCX_CI_PLAN_PATH"]
File.write(plan_path, JSON.pretty_generate(plan) + "\n") if plan_path && !plan_path.empty?

summary_path = ENV["GITHUB_STEP_SUMMARY"]
if summary_path && !summary_path.empty?
  File.open(summary_path, "a") do |summary|
    summary.puts "## FlagCX CI plan"
    summary.puts
    summary.puts "### Platform selection"
    summary.puts
    summary.puts "- Selected: **#{plan["platforms"].empty? ? "(none)" : plan["platforms"].join(", ")}**"
    routing_reason.each { |reason| summary.puts "- #{reason}" }
    summary.puts
    summary.puts "### Test plan"
    summary.puts
    summary.puts "- Stage: **#{event_name}**"
    summary.puts "- Level: **#{level}**"
    summary.puts "- Plan status: **#{plan_status}**"
    summary.puts "- Gate policy: configured exclusions stay visible; blocked work does not cancel executable jobs"
    summary.puts "- Changed files: **#{files.length}**"
    summary.puts "- Planned groups: **#{planned_groups.empty? ? "common checks only" : planned_groups.join(", ")}**"
    summary.puts "- Unit suites: **#{unit_suites.values.flatten.uniq.length} configured**"
    matrix_entries.each { |entry| summary.puts "- #{entry.fetch("platform")} selected suites: #{entry.fetch("suites").join(", ")}" }
    unit_suites.each { |platform, suites| summary.puts "- #{platform} suites: #{suites.join(", ")}" }
    summary.puts
    summary.puts "### ignore_test"
    summary.puts
    if ignore_entries.empty?
      summary.puts "- None"
    else
      ignore_entries.each do |entry|
        selector = entry.reject { |key, _value| %w[platform kind reason].include?(key) }
        summary.puts "- **#{entry["platform"]}/#{entry["kind"]}/#{entry["name"]}:** #{entry["reason"]} (selector: `#{JSON.generate(selector)}`)"
      end
    end
    summary.puts
    summary.puts "### Blocked work"
    summary.puts
    summary.puts "- None. If a workload is unavailable at runtime, record it as blocked and continue other planned tests."
  end
end

output_path = ENV["GITHUB_OUTPUT"]
abort "GITHUB_OUTPUT is required" if output_path.nil? || output_path.empty?
outputs = {
  "active" => active.to_s,
  "matrix" => JSON.generate(matrix),
  "platforms" => JSON.generate(selected.map { |entry| entry.fetch("platform") }),
  "platform_csv" => selected.map { |entry| entry.fetch("platform") }.join(","),
  "suites" => JSON.generate(matrix_entries.to_h { |entry| [entry.fetch("platform"), entry.fetch("suites")] }),
  "suite_filter_active" => suite_filter_active.to_s,
  "has_platforms" => (!selected.empty?).to_s,
  "repository" => context.fetch("repository", ""),
  "ref" => context.fetch("ref", ""),
  "level" => level,
  "plan_status" => plan_status,
  "plan_path" => plan_path.to_s
}
File.open(output_path, "a") { |output| outputs.each { |name, value| output.puts "#{name}=#{value}" } }
