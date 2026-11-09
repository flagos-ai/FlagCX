#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "yaml"

config_dir = ARGV.fetch(0, ".github/configs")
selection = ARGV.fetch(1, ENV.fetch("FLAGCX_CI_PLATFORM", "all"))
registry_path = File.join(config_dir, "platforms.yml")

load_yaml = lambda do |path|
  contents = File.read(path)
  begin
    YAML.safe_load(
      contents,
      permitted_classes: [],
      permitted_symbols: [],
      aliases: false,
      filename: path
    )
  rescue ArgumentError
    # Compatibility with the older Ruby/Psych available on some self-hosted
    # runners, whose safe_load API only accepts positional arguments.
    YAML.safe_load(contents, [], [], false, path)
  end
end

registry = if File.file?(registry_path)
             load_yaml.call(registry_path)
           else
             {}
           end
registered_platforms = registry.fetch("platforms", {})
abort "#{registry_path}: platforms must be a map" unless registered_platforms.is_a?(Hash)

selected_platforms = if selection.nil? || selection.empty? || selection == "all"
                       if registered_platforms.empty?
                         nil
                       else
                         registered_platforms.select do |_name, settings|
                           !settings.is_a?(Hash) || settings.fetch("enabled", true)
                         end.keys
                       end
                     else
                       selection.split(",").map(&:strip).reject(&:empty?)
                     end

config_files = if selected_platforms.nil?
                 Dir.glob(File.join(config_dir, "*.yml")).reject do |path|
                   File.basename(path) == "platforms.yml"
                 end.sort
               else
                 selected_platforms.map { |platform| File.join(config_dir, "#{platform}.yml") }
               end
abort "No platform configs found in #{config_dir}" if config_files.empty?

platforms = config_files.map do |path|
  abort "Platform config not found: #{path}" unless File.file?(path)

  config = load_yaml.call(path)
  required_keys = %w[hardware_name display_name]
  missing = required_keys.reject { |key| config.key?(key) }
  abort "#{path}: missing required keys: #{missing.join(', ')}" unless missing.empty?

  platform = File.basename(path, ".yml")
  hardware_name = config.fetch("hardware_name")
  abort "#{path}: hardware_name must match file name #{platform}" unless hardware_name == platform

  {
    "platform" => platform,
    "display_name" => config.fetch("display_name")
  }
end

duplicates = platforms.group_by { |platform| platform.fetch("platform") }
                      .select { |_name, entries| entries.length > 1 }
abort "Duplicate platform configs: #{duplicates.keys.join(', ')}" unless duplicates.empty?

matrix = JSON.generate({ "include" => platforms })
puts matrix

if ENV["GITHUB_OUTPUT"] && !ENV["GITHUB_OUTPUT"].empty?
  File.open(ENV.fetch("GITHUB_OUTPUT"), "a") do |output|
    output.puts "matrix=#{matrix}"
  end
end
