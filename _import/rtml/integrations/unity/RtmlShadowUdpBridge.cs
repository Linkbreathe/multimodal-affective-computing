using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public sealed class RtmlShadowUdpBridge : MonoBehaviour
{
    [SerializeField] private string participantId = "P000";
    [SerializeField] private string currentCondition = "C1";
    [SerializeField] private int pythonListenPort = 5055;
    [SerializeField] private int unityListenPort = 5056;

    private UdpClient sender;
    private UdpClient receiver;
    private Thread receiveThread;
    private volatile bool running;
    private string latestShadowMessage;

    [Serializable]
    private class UnitySample
    {
        public string schema_version = "1.0.0";
        public long unix_time_ms;
        public string participant_id;
        public string condition;
        public HeadPose head_pose;
        public EyeSample eye;
        public string frame_relative_path;
    }

    [Serializable]
    private class HeadPose
    {
        public float head_position_x, head_position_y, head_position_z;
        public float head_rotation_x, head_rotation_y, head_rotation_z, head_rotation_w;
        public float head_angular_velocity_deg_s;
    }

    [Serializable]
    private class EyeSample
    {
        public float gaze_direction_x, gaze_direction_y, gaze_direction_z;
        public bool gaze_on_painting;
    }

    private void OnEnable()
    {
        sender = new UdpClient();
        receiver = new UdpClient(new IPEndPoint(IPAddress.Loopback, unityListenPort));
        running = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true, Name = "RTML Shadow UDP" };
        receiveThread.Start();
    }

    public void ReportConditionChange(string condition)
    {
        currentCondition = condition;
        SendSample(null, null, null);
    }

    public void SendSample(Transform hmd, Vector3? gazeDirection, string frameRelativePath)
    {
        var sample = new UnitySample
        {
            unix_time_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            participant_id = participantId,
            condition = currentCondition,
            frame_relative_path = frameRelativePath
        };
        if (hmd != null)
        {
            sample.head_pose = new HeadPose
            {
                head_position_x = hmd.position.x, head_position_y = hmd.position.y, head_position_z = hmd.position.z,
                head_rotation_x = hmd.rotation.x, head_rotation_y = hmd.rotation.y,
                head_rotation_z = hmd.rotation.z, head_rotation_w = hmd.rotation.w
            };
        }
        if (gazeDirection.HasValue)
        {
            Vector3 gaze = gazeDirection.Value.normalized;
            sample.eye = new EyeSample { gaze_direction_x = gaze.x, gaze_direction_y = gaze.y, gaze_direction_z = gaze.z };
        }
        byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(sample));
        sender.Send(bytes, bytes.Length, new IPEndPoint(IPAddress.Loopback, pythonListenPort));
    }

    private void ReceiveLoop()
    {
        var endpoint = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try { latestShadowMessage = Encoding.UTF8.GetString(receiver.Receive(ref endpoint)); }
            catch (SocketException) when (!running) { }
            catch (ObjectDisposedException) { }
        }
    }

    private void Update()
    {
        if (string.IsNullOrEmpty(latestShadowMessage)) return;
        Debug.Log($"RTML Shadow suggestion (not applied): {latestShadowMessage}");
        latestShadowMessage = null;
    }

    private void OnDisable()
    {
        running = false;
        receiver?.Close();
        sender?.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join(250);
    }
}

