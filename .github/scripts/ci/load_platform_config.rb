#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "yaml"

platform = ARGV.fetch(0) { abort "Usage: #{$PROGRAM_NAME} <platform>" }
config_path = ".github/configs/#{platform}.yml"
abort "Platform config not found: #{config_path}" unless File.file?(config_path)

contents = File.read(config_path)
config = begin
  YAML.safe_load(
    contents,
    permitted_classes: [],
    permitted_symbols: [],
    aliases: false,
    filename: config_path
  )
rescue ArgumentError
  YAML.safe_load(contents, [], [], false, config_path)
end

required_keys = %w[
  hardware_name
  display_name
  ci_image
  runner_labels
  docker_volumes
  docker_options
  set_env
  unit_test_suites
]
missing = required_keys.reject { |key| config.key?(key) }
abort "#{config_path}: missing required keys: #{missing.join(', ')}" unless missing.empty?

hardware_name = config.fetch("hardware_name")
abort "#{config_path}: hardware_name must be #{platform}" unless hardware_name == platform

runner_labels = config.fetch("runner_labels")
abort "#{config_path}: runner_labels must be a non-empty array" unless runner_labels.is_a?(Array) && !runner_labels.empty?

docker_volumes = config.fetch("docker_volumes")
abort "#{config_path}: docker_volumes must be an array" unless docker_volumes.is_a?(Array)

docker_options = config.fetch("docker_options")
abort "#{config_path}: docker_options must be a non-empty array" unless docker_options.is_a?(Array) && !docker_options.empty?

validate_argument = lambda do |argument, field|
  unless argument.is_a?(String) && !argument.empty? && !argument.match?(/[\r\n\0]/)
    abort "#{config_path}: #{field} entries must be non-empty, single-line strings"
  end
end

docker_options.each { |option| validate_argument.call(option, "docker_options") }
docker_volumes.each { |volume| validate_argument.call(volume, "docker_volumes") }

docker_args = docker_options + docker_volumes.map { |volume| "--volume=#{volume}" }

suites = config.fetch("unit_test_suites")
abort "#{config_path}: unit_test_suites must be a non-empty array" unless suites.is_a?(Array) && !suites.empty?

set_env = config.fetch("set_env")
abort "#{config_path}: set_env does not exist: #{set_env}" unless File.file?(set_env)

outputs = {
  "display_name" => config.fetch("display_name"),
  "ci_image" => config.fetch("ci_image"),
  "runs_on" => JSON.generate(runner_labels),
  "docker_args" => docker_args.join("\n"),
  "set_env" => set_env,
  "unit_test_suites" => JSON.generate(suites)
}

if ENV["GITHUB_OUTPUT"] && !ENV["GITHUB_OUTPUT"].empty?
  File.open(ENV.fetch("GITHUB_OUTPUT"), "a") do |output|
    outputs.each do |key, value|
      delimiter = "FLAGCX_#{key.upcase}"
      output.puts "#{key}<<#{delimiter}"
      output.puts value
      output.puts delimiter
    end
  end
else
  puts JSON.pretty_generate(outputs)
end
