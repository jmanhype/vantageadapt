import { Stack, Card, Text, Group, Alert } from "@mantine/core";
import { useState, useEffect } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config";

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  active_strategies: number;
}

export default function SystemStatus() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get<SystemMetrics>(`${API_BASE_URL}/api/system/metrics`);
        setMetrics(response.data);
        setError(null);
      } catch (err) {
        console.error("Error:", err);
        setError("Failed to fetch system metrics");
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <Alert color="red" title="Error">
        {error}
      </Alert>
    );
  }

  if (!metrics) {
    return <Text>Loading system metrics...</Text>;
  }

  return (
    <Stack gap="md">
      <Card withBorder>
        <Text fw={500} mb="md">System Status</Text>
        <Group gap="xl">
          <Text>CPU: {metrics.cpu_usage.toFixed(1)}%</Text>
          <Text>Memory: {metrics.memory_usage.toFixed(1)}%</Text>
          <Text>Active Strategies: {metrics.active_strategies}</Text>
        </Group>
      </Card>
    </Stack>
  );
} 