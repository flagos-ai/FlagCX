#!/usr/bin/env ruby
# frozen_string_literal: true
# Resolve the platform matrix for push, PR, dispatch, and /test comment events.
# An empty PR diff skips hardware jobs; shared and unclassified changes use the
# full enabled matrix so a new or generic source path cannot bypass coverage.
# Usage: resolve_ci_matrix.rb <config-dir> <context.json>
# Writes active, matrix, platforms, repository, and ref outputs to
# GITHUB_OUTPUT. The context JSON is assembled by the controller workflow.

require "json"
require "yaml"

config_dir = ARGV.fetch(0) { abort "Usage: #{$PROGRAM_NAME} <config-dir> <context.json>" }
context_path = ARGV.fetch(1) { abort "Usage: #{$PROGRAM_NAME} <config-dir> <context.json>" }
context = JSON.parse(File.read(context_path))

load_yaml = lambda do |path|
  contents = File.read(path)
  begin
    YAML.safe_load(contents, permitted_classes: [], permitted_symbols: [], aliases: false, filename: path)
  rescue ArgumentError
    YAML.safe_load(contents, [], [], false, path)
  end
end

registry_path = File.join(config_dir, "platforms.yml")
abort "Platform registry not found: #{registry_path}" unless File.file?(registry_path)
registry = load_yaml.call(registry_path)
settings = registry.fetch("platforms", {})
abort "#{registry_path}: platforms must be a map" unless settings.is_a?(Hash)
platforms = settings.filter_map do |name, value|
  next if value.is_a?(Hash) && value.fetch("enabled", true) == false

  config_path = File.join(config_dir, "#{name}.yml")
  abort "Platform config not found: #{config_path}" unless File.file?(config_path)
  config = load_yaml.call(config_path)
  abort "#{config_path}: hardware_name must be #{name}" unless config["hardware_name"] == name
  patterns = config.fetch("paths_watch", [])
  abort "#{config_path}: paths_watch must be a non-empty array" unless patterns.is_a?(Array) && !patterns.empty?

  {
    "platform" => name,
    "display_name" => config.fetch("display_name"),
    "patterns" => patterns
  }
end
platform_names = platforms.map { |entry| entry.fetch("platform") }
abort "No enabled platform configs found in #{config_dir}" if platforms.empty?

selection = nil
active = true
event_name = context.fetch("event_name")
labels = Array(context["labels"]).map(&:to_s)
files = Array(context["files"]).map(&:to_s)

match_paths = lambda do |patterns, paths|
  patterns.any? do |pattern|
    paths.any? do |path|
      File.fnmatch?(pattern, path, File::FNM_PATHNAME | File::FNM_EXTGLOB)
    end
  end
end

select_paths = lambda do
  shared_patterns = registry.fetch("shared_paths_watch", [])
  abort "#{registry_path}: shared_paths_watch must be an array" unless shared_patterns.is_a?(Array)
  next [] if files.empty?
  next platform_names if match_paths.call(shared_patterns, files)

  selected_names = []
  files.each do |path|
    affected = platforms.filter_map do |entry|
      entry.fetch("platform") if match_paths.call(entry.fetch("patterns"), [path])
    end
    # Even one unknown file makes the diff shared, including mixed diffs.
    if affected.empty?
      selected_names = platform_names
      break
    end
    selected_names.concat(affected)
  end
  selected_names.uniq
end

parse_selection = lambda do |value|
  requested = value.to_s.split(",").map(&:strip).reject(&:empty?)
  requested = ["all"] if requested.empty?
  unknown = requested.reject { |name| name == "all" || platform_names.include?(name) }
  abort "Unknown platform selection: #{unknown.join(', ')}" unless unknown.empty?
  requested.include?("all") ? platform_names : requested.uniq
end

case event_name
when "pull_request"
  route_labels = labels.select { |label| label.start_with?("platform:") }
  invalid_labels = route_labels.map { |label| label.delete_prefix("platform:") }
                                .reject { |name| name == "all" || platform_names.include?(name) }
  abort "Unknown platform label(s): #{invalid_labels.join(', ')}" unless invalid_labels.empty?
  if route_labels.include?("platform:all")
    selection = platform_names
  elsif !route_labels.empty?
    selection = route_labels.map { |label| label.delete_prefix("platform:") }.uniq
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
  elsif command[1].nil?
    route_labels = labels.select { |label| label.start_with?("platform:") }
    selection = route_labels.empty? ? select_paths.call : parse_selection.call(route_labels.map { |label| label.delete_prefix("platform:") }.join(","))
  else
    selection = parse_selection.call(command[1])
  end
when "workflow_dispatch"
  selection = parse_selection.call(context.fetch("selection", "all"))
when "push", "schedule"
  selection = platform_names
else
  abort "Unsupported event: #{event_name}"
end

selected = platforms.select { |entry| selection.include?(entry.fetch("platform")) }
matrix = { "include" => selected.map { |entry| entry.slice("platform", "display_name") } }
outputs = {
  "active" => active.to_s,
  "matrix" => JSON.generate(matrix),
  "platforms" => JSON.generate(selected.map { |entry| entry.fetch("platform") }),
  "platform_csv" => selected.map { |entry| entry.fetch("platform") }.join(","),
  "has_platforms" => (!selected.empty?).to_s,
  "repository" => context.fetch("repository", ""),
  "ref" => context.fetch("ref", "")
}

output_path = ENV["GITHUB_OUTPUT"]
abort "GITHUB_OUTPUT is required" if output_path.nil? || output_path.empty?
File.open(output_path, "a") do |output|
  outputs.each do |name, value|
    output.puts "#{name}=#{value}"
  end
end
